"""Compare generated decoder tokens with independently executed Rust WASM."""
from argparse import ArgumentParser
import json
import struct
from time import perf_counter_ns
from unittest.mock import patch

import torch

from audit_causal_assembly import reference
from causal_wasm_bridge import HALT, SLOTS, ZERO, lower
from frozen_call_checkpoint import digest
from replay_decoder_execution_loop import run
from replay_frozen_calls import OUT as CALLS, run as run_frozen
from replay_frozen_recurrent import PACKED, signed, wasmtime_run
from replay_sparse_checkpoint import admit, expected_block, kv_bytes, splice, word_frame
from replay_word_decoder import audit
from sparse_checkpoint import OUT as SPARSE
from visuaml.compiler_lab.frozen_calls_v58 import load_call_image
from word_decoder import FRAME, HALT as HALT_TOKEN

OUT = SPARSE.parent / "causal-wasm-bridge"


def decode(model, code, words, context, first_cut=None):
    program, prefix = admit(code, words)
    memory = words
    states, segments = [], []
    while len(states) < 4096:
        # Upper bound on live nonzero words, including any address this code can store.
        capacity = len({i for i, word in enumerate(words) if word} |
                       {arg for op, _, arg in code if op == 2})
        budget = ((context-len(prefix))//FRAME-capacity-1)*FRAME
        if not segments and first_cut is not None:
            budget = min(budget, first_cut*FRAME)
        if budget <= 0:
            raise ValueError("No complete instruction fits before checkpoint reserve")
        tokens, prefill, duration, cache = run(model, prefix, True, budget=budget,
                                                halt_token=HALT_TOKEN)
        chunk, _ = audit(tokens, memory)
        states.extend(chunk)
        memory = chunk[-1][4:]
        checkpoint, checkpoint_ns, rebuild_ns = None, 0, 0
        if tokens[-1] != HALT_TOKEN:
            start = perf_counter_ns()
            checkpoint = model.checkpoint(cache, tokens[-1], SLOTS, capacity)
            checkpoint_ns = perf_counter_ns()-start
            start = perf_counter_ns()
            prefix = splice(program, checkpoint, SLOTS, capacity)
            rebuild_ns = perf_counter_ns()-start
            assert torch.equal(checkpoint, expected_block(chunk[-1]))
        segments.append(dict(tokens=tokens, checkpoint=None if checkpoint is None else checkpoint.tolist(),
                             active=cache[0][0].size(1), kv_bytes=kv_bytes(cache), prefill_ns=prefill,
                             decode_ns=duration, checkpoint_ns=checkpoint_ns, splice_ns=rebuild_ns))
        if tokens[-1] == HALT_TOKEN:
            return states, segments
    raise ValueError("Bridge replay exceeded audit budget")


def check_groups(states, initial, sources, original):
    # The stack exists on the WASM reference. The lowering exposes one
    # accumulator and only promises equivalence at complete source groups.
    memory = initial[:]
    pc = 0
    observed = []
    for state in states:
        start, end = sources[pc]
        following = state[0]
        if not state[1] or following <= pc or (start, end) != tuple(sources[following]):
            observed.append((end-1, state[4:]))
        pc = following
    endpoints = {end-1 for _, end in sources}
    expected = []
    for step in original:
        access = step["access"]
        # Use the reference operand and address from the store access.
        if step["opcode"] == "i32.store":
            assert access[1] % 4 == 0
            memory[access[1]//4] = access[3] & 0xffffffff
        if step["pc"] in endpoints:
            expected.append((step["pc"], memory[:]))
    assert observed == expected, (observed[:4], expected[:4])


def case(model, machine, guest, words, context=512, first_cut=None, padding=0):
    raw = (OUT / (guest+".wasm")).read_bytes()
    code, sources, names = lower(raw)
    if padding:
        code += [[HALT, 0, 0]]*padding
    assert words[ZERO] == 0
    page = struct.pack("<15I", *words[:ZERO]) + bytes(65536-4*ZERO)
    expected_return, expected_page = wasmtime_run(raw, 0, page)
    with patch(PACKED, side_effect=AssertionError("Packed runner entered")):
        frozen, steps = run_frozen(machine, load_call_image(raw), 0, page, trace=True)
    assert int(frozen.vm.status) == 1 and frozen.vm.page.numpy().tobytes() == expected_page

    states, segments = decode(model, code, words, context, first_cut)
    assert states == reference(code, words, 0, 4096)
    check_groups(states, words, sources, steps)
    state = states[-1]
    result = signed(state[2])
    final_page = struct.pack("<15I", *state[4:4+ZERO]) + page[4*ZERO:]
    assert result == expected_return == signed(frozen.vm.stack[0])
    assert final_page == expected_page
    return dict(guest=guest, wasm_sha256=digest(raw), source_steps=len(steps), source_opcodes=names,
                source_map=sources, initial=words, code=code, page_sha256=digest(final_page),
                result=result, states=states, segments=segments, context=context,
                generated=sum(len(s["tokens"]) for s in segments), peak=max(s["active"] for s in segments))


def admission_and_cache_controls(model, passage):
    existing = OUT.parent / "voxel-call-checkpoint" / "passage.wasm"
    try:
        lower(existing.read_bytes())
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError("Multi-function guest admitted by the bridge")
    for entry in passage:
        _, prefix = admit(entry["code"], entry["initial"])
        cached, _, _, _ = run(model, prefix, True, budget=192, halt_token=HALT_TOKEN)
        full, _, _, _ = run(model, prefix, False, budget=192, halt_token=HALT_TOKEN)
        assert cached == full and full[-1] == HALT_TOKEN
        assert audit(full, entry["initial"])[0] == entry["states"]
    edit = passage[1]
    program, prefix = admit(edit["code"], edit["initial"])
    segment = edit["segments"][0]
    partial = torch.tensor(segment["tokens"][:-1])
    _, cache = model(torch.cat((prefix, partial)), [])
    try:
        model.checkpoint(cache, int(partial[-1]), SLOTS, 4)
    except torch.jit.Error:
        pass
    else:
        raise AssertionError("Unfinished store exported")
    block = segment["checkpoint"]
    # Restore only the removed material in the emitted checkpoint.
    changed = torch.tensor(block[:-FRAME] + word_frame(2, 7) + block[-FRAME:])
    resumed = splice(program, changed, SLOTS, 4)
    tokens, _, _, _ = run(model, resumed, True, budget=192, halt_token=HALT_TOKEN)
    words = edit["states"][3][4:].copy()
    words[2] = 7
    substituted, _ = audit(tokens, words)
    assert (edit["states"][4][2], substituted[0][2]) == (0, 7)
    assert (edit["states"][5][0], substituted[1][0]) == (8, 11)
    assert substituted[-1][2] == 0 and substituted[-1][1] == 0
    return dict(original_helper_rejection=rejected, full_prefix_matches=2,
                unfinished_store_rejected=True,
                substituted=dict(checkpoint_word=2, value=7, tokens=tokens,
                                 first_changed_register=substituted[0][2],
                                 next_pc_after_branch=substituted[1][0], result=substituted[-1][2]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    artifact = SPARSE / "decoder.pt"
    model = torch.jit.load(str(artifact))
    machine = torch.jit.load(str(CALLS / "machine.pt"))
    with torch.inference_mode():
        passage = [case(model, machine, "passage", [0, edit, 7]+[0]*13,
                        first_cut=4) for edit in (0, 1)]
        assert [run["result"] for run in passage] == [0, 1]
        assert passage[1]["segments"][0]["checkpoint"] == expected_block(
            passage[1]["states"][3]).tolist() and passage[1]["states"][3][6] == 0
        cases = passage + [case(model, machine, "countdown", [0, count]+[0]*14)
                           for count in (4, 32, 128)]
        cases += [case(model, machine, "countdown", [0, 32]+[7]*live+[0]*(14-live))
                  for live in (2, 6)]
        cases += [case(model, machine, "countdown", [0, 32]+[0]*14, padding=padding)
                  for padding in (4, 9)]
        cases += [case(model, machine, "countdown", [0, 32]+[0]*14, context=256)]
        controls = admission_and_cache_controls(model, passage)
    result = dict(decoder_sha256=digest(artifact.read_bytes()),
                  frozen_vm_sha256=digest((CALLS / "machine.pt").read_bytes()),
                  controls=controls, runs=cases)
    path = OUT / "replay.json"
    if args.record:
        path.write_text(json.dumps(result, separators=(",", ":"))+"\n")
    else:
        saved = json.loads(path.read_text())
        for archive, actual in zip(saved["runs"], cases, strict=True):
            for run_case in (archive, actual):
                for segment in run_case["segments"]:
                    for key in ("prefill_ns", "decode_ns", "checkpoint_ns", "splice_ns"):
                        segment.pop(key)
        assert saved == result
    for entry in cases:
        print(entry["guest"], entry["source_steps"], entry["generated"], entry["peak"],
              len(entry["segments"])-1, entry["result"])


if __name__ == "__main__":
    main()
