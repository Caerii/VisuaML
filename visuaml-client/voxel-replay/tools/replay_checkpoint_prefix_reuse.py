"""Compare checkpoint equivalence and prefill using only immutable CODE KV."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from decoder_emitted_checkpoint import OUT as EMITTED, CHECKPOINT_TOKENS
from frozen_call_checkpoint import digest
from replay_decoder_emitted_checkpoint import splice, snapshot_after
from replay_decoder_execution_loop import run
from replay_rolling_context import OUT as ROLLING, first_diff
from replay_word_decoder import audit, pack
from word_decoder import CODE, COMMIT, FRAME, HALT, PAD

OUT = EMITTED.parent / "checkpoint-prefix-reuse"


class CodeCache:
    """One artifact and its exact, position-zero immutable CODE prefix."""

    def __init__(self, model, artifact_hash, code):
        rows = code.reshape(-1, FRAME)
        if not len(rows) or not bool((rows[:, -1] == CODE).all()):
            raise ValueError("Cache requires only immutable CODE")
        self.model = model
        self.artifact_hash = artifact_hash
        self.code = code.clone()
        self.logits, self.layers = model(self.code, [])

    def bind(self, model, artifact_hash, code, position=0):
        if model is not self.model or artifact_hash != self.artifact_hash:
            raise ValueError("Decoder identity mismatch")
        if position != 0 or not torch.equal(code, self.code):
            raise ValueError("CODE or position mismatch")
        return self.layers


def kv_bytes(cache):
    return sum(t.numel()*t.element_size() for layer in cache for t in layer)


def parity(model, code, block, bound):
    literal, full = model(torch.cat((code, block)), [])
    suffix, incremental = model(block, bound)
    assert torch.equal(literal[len(code):], suffix)
    # Equality of positions and of all layer keys and values, not only logits.
    assert all(torch.equal(x, y) for a,b in zip(full, incremental) for x,y in zip(a,b))


def rolling(model, program, initial, artifact_hash, reuse):
    begin = perf_counter_ns()
    initial_prefix = pack(program, initial)
    code = initial_prefix[:-CHECKPOINT_TOKENS]
    seed_ns = perf_counter_ns()-begin
    began_code = perf_counter_ns()
    stable = CodeCache(model, artifact_hash, code) if reuse else None
    code_ns = perf_counter_ns()-began_code if reuse else 0
    immutable = [tensor.clone() for layer in stable.layers for tensor in layer] if stable else []
    block = initial_prefix[-CHECKPOINT_TOKENS:]
    retired, trace, checkpoints, tokens_by_chunk, samples = 0, [], [], [], []
    current = initial
    peak = 0
    budget = ((512-len(initial_prefix)-CHECKPOINT_TOKENS)//FRAME)*FRAME
    if stable:
        parity(model, code, block, stable.layers)
    execution_start = perf_counter_ns()
    while retired < 1024:
        begin = perf_counter_ns()
        frame = splice(code, block, len(program))
        past = stable.bind(model, artifact_hash, frame[:-CHECKPOINT_TOKENS]) if stable else None
        validate_ns = perf_counter_ns()-begin
        tokens, prefill, decode, cache = run(model, frame if past is None else block, True,
                                            budget=budget, halt_token=HALT, prefix_cache=past)
        peak = max(peak, kv_bytes(cache))
        tokens_by_chunk.append(tokens)
        readout = 0
        if tokens[-1] != HALT:
            begin = perf_counter_ns()
            block = model.checkpoint(cache, tokens[-1])
            readout = perf_counter_ns()-begin
            checkpoints.append(block.tolist())
        states, _ = audit(tokens, current)
        trace.extend(states)
        current = states[-1][4:]
        retired += len(states)
        samples.append(dict(prefill_ns=prefill, decode_ns=decode, checkpoint_ns=readout,
                            framing_and_binding_ns=validate_ns))
        if tokens[-1] == HALT:
            break
    execution_ns = perf_counter_ns()-execution_start
    if stable:
        assert all(torch.equal(a,b) for a,b in zip(immutable, [t for pair in stable.layers for t in pair]))
    assert tokens_by_chunk[-1][-1] == HALT
    output = dict(trace=trace, checkpoints=checkpoints, tokens_by_chunk=tokens_by_chunk)
    return output, dict(seed_ns=seed_ns, code_prefill_ns=code_ns, wall_execution_ns=execution_ns,
                        total_ns=seed_ns+code_ns+execution_ns, segments=samples, active_kv_bytes=peak,
                        retained_code_kv_bytes=kv_bytes(stable.layers) if stable else 0)


def convergence(model, spec, identity):
    program = spec["paired_program"]
    samples = []
    for initial in spec["paired_inputs"]:
        expected = reference(program, initial, 0, 30)
        cut = next(i+1 for i, s in enumerate(expected) if s[0] == spec["join_pc"])
        prefix = pack(program, initial)
        stream, cache, block = snapshot_after(model, prefix, cut)
        states, writes = audit(stream, initial)
        assert states == expected[:cut]
        code = prefix[:-CHECKPOINT_TOKENS]
        resume = splice(code, block, len(program))
        stable = CodeCache(model, identity, code)
        parity(model, code, block, stable.layers)
        tail = run(model, block, True, budget=60, halt_token=HALT, prefix_cache=stable.layers)[0]
        assert tail == run(model, resume, False, budget=60, halt_token=HALT)[0]
        assert audit(tail, states[-1][4:])[0] == expected[cut:]
        samples.append(dict(initial=initial, joins_after=cut, join_state=states[-1],
                            writes=writes, history_tokens=stream, checkpoint=block.tolist(), continuation=tail))
    for other in samples[1:]:
        assert other["join_state"] == samples[0]["join_state"]
        assert other["checkpoint"] == samples[0]["checkpoint"]
        assert other["continuation"] == samples[0]["continuation"]
    assert len({len(row["history_tokens"]) for row in samples}) == 3

    base = samples[0]
    initial = base["initial"]
    prediction = spec["edit_prediction"]
    position = prediction["address"]*FRAME+7
    changed = torch.tensor(base["checkpoint"])
    changed[position:position+4] = torch.tensor(list(prediction["value"].to_bytes(4,"little")))
    state = base["join_state"].copy()
    state[4+prediction["address"]] = prediction["value"]
    code = pack(program, initial)[:-CHECKPOINT_TOKENS]
    stable = CodeCache(model, identity, code)
    result = run(model, changed, True, budget=60, halt_token=HALT, prefix_cache=stable.layers)[0]
    altered, _ = audit(result, state[4:])
    assert altered == reference(program, state[4:], 0, 5, next_pc=state[0], register=state[2])
    assert altered[0][2] == prediction["first_load_r0"] and altered[1][0] == prediction["next_branch_pc"]
    assert base["continuation"][:2] == result[:2] and base["continuation"][2] != result[2]
    return dict(histories=samples, edit=dict(state=state, checkpoint=changed.tolist(),
                first_changed_token=3, first_changed_instruction=1, trace=altered, tokens=result))


def invalid_controls(model, identity, spec):
    program = spec["paired_program"]
    prefix = pack(program, spec["paired_inputs"][0])
    code = prefix[:-CHECKPOINT_TOKENS]
    stored = CodeCache(model, identity, code)
    wrong = code.clone()
    wrong[6] = 1
    cases = {"wrong_code": (identity, wrong, 0), "wrong_position": (identity, code, FRAME),
             "wrong_artifact": ("0"*64, code, 0), "execution_suffix": (identity, prefix, 0)}
    for name,(artifact, tokens, position) in cases.items():
        try:
            stored.bind(model, artifact, tokens, position=position)
        except ValueError:
            pass
        else:
            raise AssertionError(name)

    _, _, block = snapshot_after(model, prefix, 2)
    for length in range(CHECKPOINT_TOKENS):
        try:
            splice(code, block[:length], len(program))
        except ValueError:
            pass
        else:
            raise AssertionError("Truncated checkpoint")
    for phase in range(1, FRAME):
        stream, _, _, cache = run(model, prefix, True, budget=FRAME+phase, halt_token=HALT)
        try:
            model.checkpoint(cache, stream[-1])
        except torch.jit.Error:
            pass
        else:
            raise AssertionError("Unfinished STORE")
    return dict(bind_rejections=list(cases), truncated_rejections=CHECKPOINT_TOKENS,
                unfinished_store_rejections=FRAME-1)


def replay(pair_count):
    artifact = EMITTED / "decoder.pt"
    identity = digest(artifact.read_bytes())
    model = torch.jit.load(str(artifact))
    spec = json.loads((OUT / "holdout.json").read_text())
    long = json.loads((ROLLING / "holdout.json").read_text())
    cases = {
        "rolling-519": (long["program_live_four"], [128,*long["remaining_initial_memory"]]),
        "independent": (spec["independent_program"], spec["independent_input"]),
    }
    rows, timing = {}, {}
    for key, (program, initial) in cases.items():
        prefix = pack(program, initial)
        for _ in range(2):
            run(model, prefix, True, budget=FRAME, halt_token=HALT)
        expected = reference(program, initial, 0, 1024)
        samples = {"full": [], "reuse": []}
        archived = None
        for i in range(pair_count):
            pair = {}
            for reuse in (False,True)[::1 if i%2 == 0 else -1]:
                actual, times = rolling(model, program, initial, identity, reuse)
                mismatch = first_diff(expected, actual["trace"])
                assert mismatch is None, (key,reuse,mismatch)
                pair[reuse] = actual
                samples["reuse" if reuse else "full"].append(times)
            assert pair[False] == pair[True]
            if archived is not None:
                assert pair[True] == archived
            archived = pair[True]
        code = prefix[:-CHECKPOINT_TOKENS]
        bound = CodeCache(model, identity, code)
        for block in archived["checkpoints"]:
            parity(model, code, torch.tensor(block), bound.layers)
        rows[key] = dict(program=program, initial=initial, first_divergence=None, **archived)
        timing[key] = samples
    return dict(profile="visuaml-checkpoint-prefix-reuse-v1", decoder_sha256=identity, holdout=spec,
                paired=convergence(model, spec, identity), rejection_controls=invalid_controls(model, identity, spec),
                runs=rows), timing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        bundle, samples = replay(4 if args.write else 1)
    if args.write:
        for name, data in (("replay", bundle), ("timings", samples)):
            (OUT / (name+".json")).write_text(json.dumps(data, indent=2)+"\n")
    else:
        assert bundle == json.loads((args.open or OUT / "replay.json").read_text())
    print("Paired histories, immutable CODE reuse, and rejection controls verified.")
