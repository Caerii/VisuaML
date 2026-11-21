"""Measure streaming JSONL comparison with a difference on the final row."""
import json
from tempfile import TemporaryFile
from time import perf_counter_ns
import tracemalloc

from visuaml.replay.pipeline import compare_observations, observation, read_observations


def measure(size):
    with TemporaryFile(mode="w+") as left, TemporaryFile(mode="w+") as right:
        for index in range(size):
            row = observation(str(index), "tool_result", 7, runtime="test")
            left.write(json.dumps(row)+"\n")
            if index == size-1:
                row["value"] = 9
            right.write(json.dumps(row)+"\n")
        input_bytes = [left.tell(), right.tell()]
        left.seek(0)
        right.seek(0)
        start = perf_counter_ns()
        result = compare_observations(read_observations(left), read_observations(right))
        elapsed = perf_counter_ns()-start
        left.seek(0)
        right.seek(0)
        tracemalloc.start()
        checked = compare_observations(read_observations(left), read_observations(right))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    assert checked == result and result["matched_observations"] == size-1
    return dict(rows_each=size, input_bytes=input_bytes, report_bytes=len(json.dumps(result).encode()),
                ns=elapsed, traced_python_peak_bytes=peak, first_difference=result["first_difference"])


if __name__ == "__main__":
    print(json.dumps([measure(size) for size in (1000, 100000)], indent=2))
