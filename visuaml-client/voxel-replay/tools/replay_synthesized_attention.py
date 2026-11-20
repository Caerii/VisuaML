"""Audit frozen attention, two program images and learned routing interventions."""
import argparse
import json
import struct
from unittest.mock import patch

import torch
import wasmtime

from fork_divergence import first_divergence
from frozen_call_checkpoint import digest
from internal_tool import OUT as PRIOR, InternalTool, prepare_program
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_internal_tool import oracle
from synthesize_attention_vm import OUT
from visuaml_replay import Replay


def compare_trace(expected, actual):
    index = first_divergence(expected, actual)
    if index is None:
        return None
    # Boundary index is after the instruction; PCs denote the source before it.
    pc = expected[index-1][0] if index and index <= len(expected) else 0
    return dict(boundary=index+1, source_pc=pc, expected=expected[index:index+1],
                actual=actual[index:index+1])


def router():
    torch.manual_seed(11)
    model = InternalTool()
    model.circuit = torch.jit.load(str(OUT / "executor.pt"))
    tokens = torch.tensor([[i, filler] for i in range(3) for filler in (3, 4)])
    labels = torch.arange(3).repeat_interleave(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(100):
        loss = torch.nn.functional.cross_entropy(model.logits(tokens), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.equal(model.logits(tokens).argmax(-1), labels)
    return torch.jit.freeze(torch.jit.script(model.eval()))


def store_events(logical_program, trace):
    source = 0
    stores = []
    for state in trace:
        op, _, address = logical_program[source]
        if op == 2:
            stores.append([4*address, state[4+address]])
        source = state[0]
    return stores


def audit_wasm(wat, operands, expected, stores):
    raw = bytes(wasmtime.wat2wasm(wat))
    replay = Replay(raw)
    memory = bytearray(65536)
    struct.pack_into("<I", memory, 0, operands[0])
    from visuaml.compiler_lab.frozen_calls_v58 import initialize
    state = initialize(replay.image, operands[1], memory, 2000)
    observed = []
    while int(state.vm.status) == 0:
        pulse = replay.graph(replay.image.rows, state)
        state = pulse.state
        kind, address, _, data = pulse.access.tolist()
        if kind == 2:
            observed.append([address, data])
    final = state.vm.page.numpy().tobytes()
    result, page = wasmtime_run(raw, operands[1], memory)
    assert int(state.vm.status) == 1
    assert observed == stores
    assert page == final and struct.unpack_from("<4I", final) == tuple(expected[-1][4:])
    assert result & 0xffffffff == int(state.vm.stack[0]) == expected[-1][4]
    return dict(wasm_sha256=digest(raw), page_sha256=digest(final), ordered_stores=observed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = json.loads((OUT / "programs.json").read_text())
    path = OUT / "executor.pt"
    assert digest(path.read_bytes()) == spec["frozen_sha256"]
    model = torch.jit.load(str(path))
    integrated = router() if args.write else torch.jit.load(str(OUT / "integrated.pt"))
    records = {}
    with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        archived = json.loads((PRIOR / "replay.json").read_text())
        baseline = next(x["tool_trace"] for x in archived["cases"]
                        if x["mode"] == 1 and x["operands"] == [2, 4])
        assert oracle(spec["original"], [2, 4]) == baseline
        for key, wasm_file in (("original", "reference.wat"), ("post_freeze", "post-freeze.wat"),
                               ("final_holdout", "final-holdout.wat")):
            code = spec[key]
            image = prepare_program(code)
            cases = []
            for operands in ([2, 4], [0, 1], [4294967294, 3]):
                expected = oracle(code, operands)
                answer, trace = model(image, torch.tensor(operands))
                actual = trace[:len(expected)].tolist()
                difference = compare_trace(expected, actual)
                assert difference is None, (key, operands, difference)
                assert torch.equal(trace[len(expected):], trace[len(expected)-1].expand_as(trace[len(expected):]))
                assert int(answer) == expected[-1][4]
                mirror = audit_wasm((OUT / wasm_file).read_text(), operands, expected, store_events(code, expected))
                cases.append(dict(operands=operands, trace=actual, **mirror))
            records[key] = cases
        extremes = 0
        for seed in (0, 1, 2147483647, 2147483648, 4294967294, 4294967295):
            for immediate in (-2147483648, -5, -1, 0, 1, 2147483647):
                mini = [[1, 0, 0], [3, 0, immediate], [2, 0, 0], [0, 0, 0]]
                _, trace = model(prepare_program(mini), torch.tensor([seed, 1]))
                assert trace[:4].tolist() == oracle(mini, [seed, 1])
                extremes += 1
        code = spec["final_holdout"]
        image = prepare_program(code)
        expected = oracle(code, [2, 4])
        tampered = [row.copy() for row in expected]
        first_store = 1
        tampered[first_store][6] ^= 1
        delta = compare_trace(expected, tampered)
        assert delta["boundary"] == 2 and delta["source_pc"] == 1
        interventions = []
        for mode, values in ((0, [2, 4]), (1, [2, 4]), (2, [4, 2])):
            def invoke(enabled, perturbation):
                return integrated(torch.tensor([4, mode]), torch.tensor(values), image,
                                  torch.tensor(enabled), torch.tensor(perturbation))
            normal, route, operands, trace = invoke(True, 0)
            disabled, _, _, skipped = invoke(False, 0)
            shifted, _, _, _ = invoke(True, 9)
            assert int(route) == mode and not skipped.numel()
            assert int(disabled) == values[0]
            if mode:
                assert operands.tolist() == [2, 4] and trace[:len(expected)].tolist() == expected
                assert int(normal) == expected[-1][4] and int(shifted) == int(normal)+9
                assert int(normal) != int(disabled)
            else:
                assert int(normal) == int(disabled) == int(shifted) and not trace.numel()
            interventions.append(dict(mode=mode, inputs=values, selected=operands.tolist(),
                                      answer=int(normal), disabled=int(disabled), shifted=int(shifted)))
    # Explicit admission regression for the first illegal memory word.
    try:
        prepare_program([[2, 0, 4], [0, 0, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range write was admitted")
    evidence = dict(frozen_sha256=spec["frozen_sha256"], cases=records,
                    i32_boundary_cases=extremes, injected_divergence=delta, interventions=interventions)
    replay = OUT / "replay.json"
    if args.write:
        replay.write_text(json.dumps(evidence, indent=2)+"\n")
        integrated.save(str(OUT / "integrated.pt"))
    else:
        assert json.loads(replay.read_text()) == evidence
    print(f"Nine exact traces, {extremes} i32 edge cases; first injected mismatch after source PC {delta['source_pc']}; router interventions pass.")


if __name__ == "__main__":
    main()
