"""Count completed full-BFS regions independently of timing runs."""
import json
from collections import Counter
from unittest.mock import patch

from visuaml.compiler_lab import neural_microop_executor_v57 as vm

from operand_pairs import run, workload
from prepared_region_pairs import VARIANTS


def main():
    case = workload("bfs")
    results = {}
    options = {
        "pair": VARIANTS["region"],
        "local_region": {**VARIANTS["region"], "prepare_local_region": True},
    }
    for name, flags in options.items():
        counts = Counter()
        originals = {key: getattr(vm, key) for key in (
            "_run_const_add_region", "_run_local_const_add_region", "select_frame_action",
        )}

        def count(key):
            def tracked(*args, **kwargs):
                counts[key] += 1
                return originals[key](*args, **kwargs)
            return tracked

        with patch.multiple(vm, **{key: count(key) for key in originals}), patch(
            "visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
            side_effect=AssertionError("Unexpected packed fallback"),
        ):
            samples = []
            run(case, samples, None, prepare_operand_pop=True, prepare_alu=True, **flags)
        steps = samples[0]["steps"]
        pairs = counts["_run_const_add_region"]
        triples = counts["_run_local_const_add_region"]
        generic = counts["select_frame_action"]
        assert steps == generic + 2 * pairs + 3 * triples
        results[name] = {
            "steps": steps, "pair_calls": pairs, "triple_calls": triples,
            "individual_host_dispatches": generic, "total_host_dispatches": generic + pairs + triples,
            "full_operand_cell_copies_avoided": 3 * pairs + 4 * triples,
            "page_sha256": samples[0]["page_sha256"],
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
