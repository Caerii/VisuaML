"""Replay the vocabulary execution loop without decoding tokens in the driver."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from causal_kv_memory import OUT as KV
from causal_tensor_assembly import OUT as STRUCTURED, decode, observe, start
from decoder_execution_loop import OUT
from frozen_call_checkpoint import digest
from vocabulary_tensor_assembly import OUT as VOCABULARY, FRAME, encode_image, generate


def pack(program, memory):
    if (len(program) > 16 or not program or program[-1][0] != 0
        or len(memory) != 4 or any(bit not in (0, 1) for bit in memory)):
        raise ValueError("Invalid program or initial memory")
    tokens = []
    for pc, (op, register, arg) in enumerate(program):
        if (register != 0 or op not in (0, 1, 2, 4) or not 0 <= arg < 16
            or (op in (1, 2) and arg >= 4) or (op == 4 and arg >= len(program))
            or (op == 0 and arg != 0)):
            raise ValueError("Outside the one-bit, one-register ISA")
        tokens.extend((28, pc, 16+op, arg, 28, 25))
    for address, bit in enumerate(memory):
        tokens.extend((28, 0, 21, address, 21+bit, 26))
    return torch.tensor(tokens + [28, 0, 21, 23, 21, 26])


def run(model, prefix, incremental, budget=150, substitute=None):
    if not 0 <= budget <= 512-prefix.numel():
        raise ValueError("Token budget exceeds context")
    tokens = []
    prefill_start = perf_counter_ns()
    logits, cache = model(prefix, [])
    prefill_ns = perf_counter_ns()-prefill_start
    decode_start = perf_counter_ns()
    for _ in range(budget):
        token = int(logits[-1].argmax())
        if substitute is not None and len(tokens) == substitute[0]:
            token = substitute[1]
        tokens.append(token)
        if token == 27:
            break
        next_token = torch.tensor([token])
        if incremental:
            logits, cache = model(next_token, cache)
        else:
            prefix = torch.cat((prefix, next_token))
            logits, cache = model(prefix, [])
    # Cache includes every consumed token. The final selected HALT is a
    # terminal output and does not need to be consumed by another forward.
    return tokens, prefill_ns, perf_counter_ns()-decode_start, cache


def audit(tokens, initial):
    memory = initial.copy()
    states, writes = [], []
    for start_at in range(0, len(tokens)-5, 6):
        begin, pc, reg, address, bit, marker = tokens[start_at:start_at+6]
        assert begin == 24 and marker in (26, 27) and reg in (21, 22)
        assert address in (0, 1, 2, 3, 23) and bit in (21, 22)
        if address != 23:
            assert marker == 26
            memory[address] = bit-21
            writes.append(dict(step=len(states)+1, token=start_at+6, address=address, value=bit-21))
        states.append([pc, int(marker != 27), reg-21, 0, *memory])
    return states, writes


def replay(measure):
    path = OUT / "decoder.pt"
    model = torch.jit.load(str(path))
    structured = torch.jit.load(str(STRUCTURED / "decoder.pt"))
    hybrid = torch.jit.load(str(KV / "hybrid.pt"))
    vocabulary = torch.jit.load(str(VOCABULARY / "decoder.pt"))
    spec = json.loads((OUT / "holdout.json").read_text())
    program = spec["program"]
    rows, timings = {}, {}
    for name, initial in spec["memory_by_case"].items():
        prefix = pack(program, initial)
        stream, _, _, cache = run(model, prefix, True)
        assert stream == run(model, prefix, False)[0]
        states, writes = audit(stream, initial)
        assert states and states[-1][1] == 0
        expected = reference(program, initial, 0, len(states))
        assert states == expected
        image = start(program, initial, 0)
        for other in (observe(decode(structured, image, len(states)), initial),
                      observe(hybrid(image, len(states)), initial)):
            assert states == other
        packed = encode_image(image)
        assert states == observe(vocabulary.committed(generate(vocabulary, packed, len(states)*FRAME)), initial)
        rows[name] = dict(prefix=prefix.tolist(), tokens=stream, states=states, writes=writes,
                          cache_positions=cache[0][0].size(1),
                          cache_bytes=sum(t.numel()*t.element_size() for layer in cache for t in layer))
        if measure:
            samples = {}
            calls = {
                "cached": lambda: run(model, prefix, True),
                "full_prefix": lambda: run(model, prefix, False),
                "structured": lambda: decode(structured, image, len(states)),
                "hybrid": lambda: hybrid(image, len(states)),
                "vocabulary": lambda: generate(vocabulary, packed, len(states)*FRAME),
            }
            for call in calls.values():
                call()
            for index in range(8):
                for key in list(calls)[::1 if index % 2 == 0 else -1]:
                    began = perf_counter_ns()
                    result = calls[key]()
                    elapsed = perf_counter_ns()-began
                    item = dict(total_ns=elapsed)
                    if key in ("cached", "full_prefix"):
                        item.update(prefill_ns=result[1], decode_ns=result[2])
                    samples.setdefault(key, []).append(item)
            timings[name] = samples

    taken = rows["taken"]
    config = spec["intervention"]
    commit = next(write["token"] for write in taken["writes"] if write["step"] == config["source_step"])
    prefix = torch.tensor(taken["prefix"])
    end = len(prefix)+commit
    _, cached_after_commit = model(torch.tensor(taken["prefix"]+taken["tokens"][:commit]), [])
    key, payload = cached_after_commit[2]
    assert not key[0, end-6:end-1, :4].any() and not payload[0, end-6:end-1].any()
    assert int(key[0, end-1, config["address"]]) == 1 and int(payload[0, end-1, 0]) == 1
    # Stop after each partial store prefix, then verify that audit observes the older word.
    for partial in range(5):
        cut = commit-6+partial+1
        assert run(model, prefix, True, budget=cut)[0] == taken["tokens"][:cut]
        assert audit(taken["tokens"][:cut], spec["memory_by_case"]["taken"])[0][-1][6] == 1
        _, partial_cache = model(torch.tensor(taken["prefix"]+taken["tokens"][:cut]), [])
        assert not partial_cache[2][0][0, end-6:len(prefix)+cut, :4].any()
    substitution = (commit-2, 21+config["replacement"])
    changed = run(model, prefix, True, substitute=substitution)[0]
    assert changed == run(model, prefix, False, substitute=substitution)[0]
    altered, writes = audit(changed, spec["memory_by_case"]["taken"])
    assert altered[:3] == taken["states"][:3] and altered[3][6] == config["replacement"]
    assert altered[config["load_step"]-1][2] == config["replacement"]
    pc = altered[config["branch_step"]-1][0]
    assert pc == config["expected_next_pc"] and program[pc][0] == config["expected_next_opcode"]
    assert any(write["step"] == 7 and write["address"] == 3 for write in writes)
    assert altered[-1][1] == 0
    loop_prefix = pack(spec["context_program"], [1, 0, 0, 0])
    limit = 512-loop_prefix.numel()
    loop, _, _, full_cache = run(model, loop_prefix, True, budget=limit)
    loop_trace, _ = audit(loop, [1, 0, 0, 0])
    assert loop_trace == reference(spec["context_program"], [1, 0, 0, 0], 0, limit//6)
    full_logits, _ = model(torch.cat((loop_prefix, torch.tensor(loop))), [])
    assert full_logits[len(loop_prefix)-1:-1].argmax(dim=-1).tolist() == loop
    assert full_cache[0][0].size(1) == 512
    output = dict(context_probe=dict(tokens=loop, source_steps=len(loop_trace), context_tokens=512),
                  profile="visuaml-decoder-execution-loop-v1", decoder_sha256=digest(path.read_bytes()), holdout=spec, cases=rows,
                  intervention=dict(tokens=changed, states=altered, writes=writes, changed_token=commit-1,
                                    next_pc=pc, next_opcode=program[pc][0]), partial_store_lengths=5)
    if measure:
        (OUT / "timings.json").write_text(json.dumps(timings, indent=2)+"\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    options = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        result = replay(options.write)
    if options.write:
        (OUT / "replay.json").write_text(json.dumps(result, indent=2)+"\n")
    else:
        assert result == json.loads((options.open or OUT / "replay.json").read_text())
    print("Generated stores, cached LOAD, opposite branches and HALT verified in both modes.")
