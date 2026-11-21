"""Replay canonical sparse checkpoints, independently auditing guest state."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from decoder_emitted_checkpoint import OUT as FIXED
from frozen_call_checkpoint import digest
from replay_decoder_execution_loop import run
from replay_word_decoder import audit, pack
from sparse_checkpoint import OUT
from word_decoder import FRAME, COMMIT, CODE, HALT, NONE, PAD

CONTEXT = 512


def word_frame(address, word):
    return [PAD,0,0,0,0,0,address,*word.to_bytes(4,"little"),COMMIT]


def admit(program, memory):
    if not 1 <= len(memory) <= 16 or not 1 <= len(program) <= 16 or program[-1][0] != 0:
        raise ValueError("Invalid program or memory range")
    tokens = []
    for pc,(op, register, operand) in enumerate(program):
        if (register != 0 or op not in (0,1,2,3,4) or
            (op in (1,2) and not 0 <= operand < len(memory)) or
            (op == 4 and not 0 <= operand < len(program)) or
            (op == 3 and not -2**31 <= operand < 2**31) or (op == 0 and operand != 0)):
            raise ValueError("Outside the sparse word ABI")
        immediate = operand % 2**32 if op == 3 else 0
        tokens.extend([PAD,pc,256+op, PAD,PAD,PAD, 0 if op == 3 else operand,
                       *immediate.to_bytes(4,"little"),CODE])
    code = torch.tensor(tokens)
    for address, value in enumerate(memory):
        if value:
            tokens.extend(word_frame(address, value))
    return code, torch.tensor(tokens + word_frame(NONE,0))


def splice(code, block, address_range, capacity):
    if block.ndim != 1 or block.dtype != torch.long or len(block) % FRAME or not FRAME <= len(block) <= (capacity+1)*FRAME:
        raise ValueError("Incomplete or oversized checkpoint")
    rows = block.reshape(-1, FRAME).tolist()
    ids = [row[6] for row in rows[:-1]]
    if ids != sorted(set(ids)) or any(not 0 <= addr < address_range for addr in ids):
        raise ValueError("Noncanonical addresses")
    if any(row[0] != PAD or row[-1] != COMMIT for row in rows):
        raise ValueError("Invalid checkpoint frame")
    state = rows[-1]
    if (state[6] != NONE or not 0 <= state[1] < len(code)//FRAME or any(state[7:11])
        or any(not 0 <= byte < 256 for byte in state[2:6])):
        raise ValueError("Missing live state")
    for row in rows[:-1]:
        if any(row[1:6]) or not any(row[7:11]) or any(not 0 <= byte < 256 for byte in row[7:11]):
            raise ValueError("Noncanonical memory frame")
    return torch.cat((code, block))


def expected_block(state):
    pc, live, register, _, *memory = state
    assert live
    return torch.tensor([token for address,value in enumerate(memory) if value
                         for token in word_frame(address,value)] +
                        [PAD,pc,*register.to_bytes(4,"little"),NONE,0,0,0,0,COMMIT])


def kv_bytes(cache):
    return sum(t.numel()*t.element_size() for pair in cache for t in pair)


def timing(fn):
    start = perf_counter_ns()
    outcome = fn()
    return outcome, perf_counter_ns()-start


def long_replay(model, spec, initial):
    code, prefix = admit(spec["program"], initial)
    expected = reference(spec["program"], initial, 0, 512)
    cap, memory = spec["capacity"], initial
    trace, chunks, samples = [], [], []
    while len(trace) < 512:
        budget = ((CONTEXT-len(prefix)-FRAME*(cap+1))//FRAME)*FRAME
        assert budget > 0
        start = perf_counter_ns()
        tokens, prefill, decode, cache = run(model, prefix, True, budget=budget, halt_token=HALT)
        block, readout_ns, splice_ns = None, 0, 0
        peak = len(prefix)+len(tokens)
        if tokens[-1] != HALT:
            block, readout_ns = timing(lambda: model.checkpoint(cache, tokens[-1], len(memory), cap))
            rebuilt, splice_ns = timing(lambda: splice(code, block, len(memory), cap))
        elapsed = perf_counter_ns()-start
        states, _ = audit(tokens, memory)
        assert states == expected[len(trace):len(trace)+len(states)]
        trace.extend(states)
        memory = states[-1][4:]
        if block is not None:
            assert torch.equal(block, expected_block(states[-1]))
            assert memory[14] == 0 and memory[15] == 7
        chunks.append(dict(tokens=tokens, checkpoint=block.tolist() if block is not None else None,
                           peak_tokens=peak, kv_bytes=kv_bytes(cache), positions=cache[0][0].size(1)))
        samples.append(dict(prefill_ns=prefill, decode_ns=decode, readout_ns=readout_ns,
                            splice_ns=splice_ns, total_ns=elapsed))
        if tokens[-1] == HALT:
            break
        prefix = rebuilt
    assert trace == expected and not trace[-1][1]
    return dict(trace=trace, segments=chunks), samples


def capacity_controls(model, prefix, cache, boundary, address_range):
    checks = []
    for label, fn in {
        "overflow": lambda: model.checkpoint(cache, boundary, address_range, 1),
        "unfinished_store": lambda: model.checkpoint(model(prefix, [])[1], int(prefix[-1]), address_range, 4),
        "unreserved": lambda: model.checkpoint(cache, boundary, address_range, 16),
    }.items():
        try:
            fn()
        except torch.jit.Error as exc:
            expected = {"overflow": "Sparse checkpoint overflow", "unfinished_store": "Live COMMIT required",
                        "unreserved": "Unreserved checkpoint"}[label]
            assert expected in str(exc)
            checks.append(label)
        else:
            raise AssertionError(label)
    return checks


def convergence(model, spec):
    program, address_range = spec["program"], spec["range"]
    cap = spec["capacity"]
    histories, raw = [], []
    for count, stale in zip(spec["history_iterations"], spec["history_stale_words"]):
        initial = [0]*address_range
        initial[0], initial[1], initial[14] = count, 7, stale
        code, prefix = admit(program, initial)
        steps = 4+count*4
        tokens, _, _, cache = run(model, prefix, True, budget=steps*FRAME, halt_token=HALT)
        states, _ = audit(tokens, initial)
        assert states == reference(program, initial, 0, steps)
        state = states[-1]
        block = model.checkpoint(cache, tokens[-1], address_range, cap)
        assert torch.equal(block, expected_block(state))
        resumed = splice(code, block, address_range, cap)
        suffix = run(model, resumed, True, budget=48, halt_token=HALT)[0]
        whole = run(model, prefix, True, budget=(steps+4)*FRAME, halt_token=HALT)[0]
        assert suffix == whole[len(tokens):]
        histories.append(dict(initial=initial, tokens=tokens, join=state, block=block.tolist(), suffix=suffix))
        raw.append(micro(model, code, cache, tokens[-1], address_range, cap))
    base = histories[0]
    assert all(item["join"] == base["join"] and item["block"] == base["block"]
               and item["suffix"] == base["suffix"] for item in histories)
    assert [row[6] for row in torch.tensor(base["block"]).reshape(-1,FRAME).tolist()] == [1,15,NONE]
    # Later branch visibly depends on the old high-address value.
    prediction = spec["intervention"]
    entries = torch.tensor(base["block"]).reshape(-1,FRAME)
    edited = entries[entries[:,6] != prediction["remove_address"]].flatten()
    state = base["join"].copy()
    state[4+prediction["remove_address"]] = 0
    resumed = run(model, splice(code, edited, address_range, cap), True, budget=48, halt_token=HALT)[0]
    trace, _ = audit(resumed, state[4:])
    assert trace == reference(program, state[4:], 0, 4, next_pc=state[0], register=state[2])
    assert trace[0][2] == prediction["expected_load"] and trace[1][0] == prediction["expected_branch_pc"]
    partial, _, _, _ = run(model, prefix, True, budget=3*FRAME+11, halt_token=HALT)
    checks = capacity_controls(model, torch.cat((prefix, torch.tensor(partial))), cache, tokens[-1], address_range)
    return dict(histories=histories, intervention=dict(tokens=resumed, trace=trace), rejections=checks), raw


def micro(model, code, cache, marker, address_range, capacity, expected=None):
    for _ in range(2):
        model.checkpoint(cache, marker, address_range, capacity)
    result = []
    for _ in range(16):
        words, scan = timing(lambda: model.scan(cache, address_range))
        block, emit = timing(lambda: model.emit(cache, marker, words, capacity))
        assert torch.equal(block, model.checkpoint(cache, marker, address_range, capacity))
        if expected is not None:
            assert torch.equal(block, expected)
        prefix = splice(code, block, address_range, capacity)
        (_, fresh), prefill = timing(lambda: model(prefix, []))
        result.append(dict(scan_ns=scan, emission_ns=emit, prefill_ns=prefill,
                           checkpoint_tokens=len(block), source_positions=cache[0][0].size(1),
                           fresh_kv_bytes=kv_bytes(fresh)))
    return result


def axes(model):
    # Same source tape for the range axis; all nonzero words are within the
    # smallest declared range. For density, explicit zero COMMIT rows hold
    # old cache length fixed even when the emitted checkpoint becomes smaller.
    code, _ = admit([[0,0,0]], [0]*16)
    def cache_for(count):
        words = [0]*8
        for address in range(count):
            words[address] = 70+address
        prefix = torch.cat((code, torch.tensor([token for i,w in enumerate(words)
                              for token in word_frame(i,w)] + word_frame(NONE,0))))
        _, cache = model(prefix, [])
        return cache
    shared = cache_for(2)
    def expected(count):
        return expected_block([0,1,0,0,*[70+addr if addr < count else 0 for addr in range(16)]])

    a = {str(size): micro(model, code, shared, COMMIT, size, 8, expected(2)) for size in (8,12,16)}
    b = {str(count): micro(model, code, cache_for(count), COMMIT, 16, 8, expected(count)) for count in (0,2,4,8)}
    return dict(address_range=a, retained_words=b)


def fixed_control(model):
    original = torch.jit.load(str(FIXED / "decoder.pt"))
    program = [[1,0,1],[2,0,2],[1,0,0],[2,0,2],[1,0,2],[4,0,7],[2,0,3],[0,0,0]]
    results = []
    for seed in (0, 7):
        memory = [seed,4,0,0]
        code, sparse = admit(program, memory)
        fixed = pack(program, memory)
        a, _, _, sparse_cache = run(model, sparse, True, budget=48, halt_token=HALT)
        b, _, _, fixed_cache = run(original, fixed, True, budget=48, halt_token=HALT)
        assert a == b
        state = audit(b, memory)[0][-1]
        sparse_block = model.checkpoint(sparse_cache, a[-1], 4, 4)
        dense_block = original.checkpoint(fixed_cache, b[-1])
        assert torch.equal(sparse_block, expected_block(state))
        semantic_dense = [int.from_bytes(bytes(row[7:11]), "little")
                          for row in dense_block.reshape(-1,FRAME).tolist()[:4]]
        assert semantic_dense == state[4:]
        sparse_tail = run(model, splice(code, sparse_block, 4, 4), True, budget=48, halt_token=HALT)[0]
        dense_tail = run(original, torch.cat((code, dense_block)), True, budget=48, halt_token=HALT)[0]
        assert sparse_tail == dense_tail
        results.append(dict(seed=seed, source_steps=(len(a)+len(sparse_tail))//FRAME,
                            dense_checkpoint=dense_block.tolist(), sparse_checkpoint=sparse_block.tolist(),
                            continuation=sparse_tail))
    return results


def replay(measure):
    path = OUT / "decoder.pt"
    model = torch.jit.load(str(path))
    spec = json.loads((OUT/"holdout.json").read_text())
    initial = [0]*spec["range"]
    for key,value in spec["initial_entries"].items():
        initial[int(key)] = value
    histories, history_times = convergence(model, spec)
    execution, long_times = long_replay(model, spec, initial)
    samples = dict(history=history_times, execution=long_times, axes=axes(model) if measure else {})
    return dict(profile="visuaml-sparse-checkpoint-v1", decoder_sha256=digest(path.read_bytes()),
                holdout=spec, histories=histories, execution=execution, fixed_reference=fixed_control(model)), samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    options = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        bundle, raw = replay(options.write)
    if options.write:
        for name,data in (("replay",bundle), ("timings",raw)):
            (OUT/(name+".json")).write_text(json.dumps(data, indent=2)+"\n")
    else:
        assert bundle == json.loads((options.open or OUT/"replay.json").read_text())
    print("Verified sparse continuation, convergence, overflow, default erasure and four-word parity.")
