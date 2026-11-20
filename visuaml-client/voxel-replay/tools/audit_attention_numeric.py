"""Verify the shipped coefficients and bounded binary64 transition contract."""
import json

import torch

from frozen_call_checkpoint import digest
from internal_tool import prepare_program
from replay_internal_tool import oracle
from synthesize_attention_vm import OUT as EXECUTOR, SynthesizedVM
from two_call_controller import OUT


def main():
    torch.set_num_threads(1)
    source = SynthesizedVM()
    exported = json.loads((EXECUTOR / "weights.json").read_text())["weights"]
    assert {name: value.tolist() for name, value in source.named_buffers()} == exported
    compiled = EXECUTOR / "executor.pt"
    original = json.loads((EXECUTOR / "programs.json").read_text())
    assert digest(compiled.read_bytes()) == original["frozen_sha256"]
    model = torch.jit.load(str(compiled))
    bounded = torch.jit.freeze(torch.jit.script(source.eval()))
    cases = []
    for seed in (0, 2147483647, 2147483648, 4294967294, 4294967295):
        for immediate in (-2147483648, -1, 0, 3, 2147483647):
            code = [[1, 0, 0], [3, 0, immediate], [2, 0, 0], [3, 1, -1], [4, 1, 0], [0, 0, 0]]
            operands = [seed, 9]
            expected = oracle(code, operands)
            answer, trace = model(prepare_program(code), torch.tensor(operands))
            partial_answer, partial_trace = bounded(prepare_program(code), torch.tensor(operands), 46)
            assert int(partial_answer) == int(answer) and partial_trace.tolist() == expected
            assert len(expected) == 46 and trace[:46].tolist() == expected
            assert not int(trace[-1, 1]) and int(answer) == expected[-1][4]
            cases.append(dict(seed=seed, signed_immediate=immediate, retired=46,
                              first_store=expected[2][4], final_word=int(answer)))
    _, trace = model(prepare_program(original["original"]), torch.tensor([4294967294, 10]))
    assert int(trace[-1, 1]) == 1
    bound = source.layout["affine_magnitude_bound"]
    facts = dict(executor_sha256=original["frozen_sha256"], dtype="binary64",
                 max_affine_absolute_sum_bound=bound, integer_exactness_limit=2**53,
                 fetch_score_absolute_sum_bound=2*47*47+47*47,
                 unused_memory_head_score_absolute_sum_bound=6*2**31+9,
                 tested_cases=cases, stopped_with_live_flag_after_cycles=48)
    (OUT / "numeric.json").write_text(json.dumps(facts, indent=2)+"\n")
    print(f"Affine row bound: {bound}; 25 full 46-step boundary traces and exhaustion checked.")


if __name__ == "__main__":
    main()
