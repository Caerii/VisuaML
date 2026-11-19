"""Include the first library build in a sustained exact terrain replay."""
import json
from time import perf_counter_ns
from unittest.mock import patch

from compiled_frame_event_pairs import SETTINGS
from operand_pairs import run
from terrain_verify import scale_session


def main():
    session = scale_session(8, False, 16)
    samples = {}
    expected = None
    with patch("visuaml.compiler_lab.neural_microstep_executor_v58._NativeMicrostep",
               side_effect=AssertionError("Unexpected packed fallback")):
        for name, enabled in (("numpy", False), ("compiled_cold", True)):
            rows = []
            start = perf_counter_ns()
            expected = run(session, rows, expected, **SETTINGS, use_compiled_frame_event=enabled)
            samples[name] = {"wall_ns": perf_counter_ns() - start, "ticks": rows}
    completed = []
    cumulative = {name: 0 for name in samples}
    for i in range(len(session["inputs"])):
        for name, value in samples.items():
            row = value["ticks"][i]
            cumulative[name] += row["pre_execution_ns"] + row["execution_ns"]
        completed.append(dict(cumulative))
    print(json.dumps({"session": session, "samples": samples,
                      "cumulative_executor_ns": completed,
                      "first_ticks_ahead": next((i + 1 for i, row in enumerate(completed)
                                                 if row["compiled_cold"] < row["numpy"]), None),
                      "verified_tick_hashes": [tick["page_sha256"] for tick in expected["ticks"]]},
                     indent=2))


if __name__ == "__main__":
    main()
