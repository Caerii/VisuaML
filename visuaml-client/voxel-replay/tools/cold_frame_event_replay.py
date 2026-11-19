"""Include the first library build in a sustained exact terrain replay."""
import argparse
import json
from time import perf_counter_ns
from unittest.mock import patch

from compiled_frame_event_pairs import SETTINGS
from operand_pairs import run
from terrain_verify import scale_session
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    session = scale_session(8, False, 16)
    samples = {}
    expected = None
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        configurations = (("off", True), ("copy", True), ("resident", True)) if args.fused else (
            ("numpy", False), ("compiled_cold", True),
        )
        for name, enabled in configurations:
            compiled_alu.cache_clear()
            rows = []
            start = perf_counter_ns()
            expected = run(session, rows, expected, **SETTINGS, use_compiled_frame_event=enabled,
                           compiled_frame_retain=name if args.fused else "off")
            samples[name] = {"wall_ns": perf_counter_ns() - start, "ticks": rows}
    completed = []
    cumulative = {name: 0 for name in samples}
    for i in range(len(session["inputs"])):
        for name, value in samples.items():
            row = value["ticks"][i]
            cumulative[name] += row["pre_execution_ns"] + row["execution_ns"]
        completed.append(dict(cumulative))
    reference = next(iter(samples))
    first = {name: next((i + 1 for i, row in enumerate(completed) if row[name] < row[reference]), None)
             for name in samples if name != reference}
    print(json.dumps({"session": session, "samples": samples, "reference": reference,
                      "cumulative_executor_ns": completed, "first_ticks_ahead": first,
                      "verified_tick_hashes": [tick["page_sha256"] for tick in expected["ticks"]]},
                     indent=2))


if __name__ == "__main__":
    main()
