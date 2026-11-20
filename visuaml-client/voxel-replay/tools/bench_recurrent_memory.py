"""Measure the frozen recurrent machine's indexed memory primitives on CPU."""
import json
import statistics
from time import perf_counter_ns

import torch


def main():
    torch.set_num_threads(1)
    samples = 31
    repetitions = 64
    results = []
    for size in (4096, 65536, 1048576, 16777216):
        page = torch.zeros(size, dtype=torch.uint8)
        indices = torch.arange(size // 2 + 1, size // 2 + 5)
        value = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        def read():
            return torch.gather(page, 0, indices)

        def write():
            return torch.scatter(page, 0, indices, value)

        assert torch.equal(read(), torch.zeros_like(value))
        updated = write()
        assert torch.equal(torch.gather(updated, 0, indices), value)
        assert torch.equal(read(), torch.zeros_like(value))
        del updated
        for _ in range(repetitions):
            read()
            write()
        timings = {"gather": [], "functional_scatter": []}
        pair = (("gather", read), ("functional_scatter", write))
        for sample in range(samples):
            for name, operation in pair[::(-1 if sample % 2 else 1)]:
                start = perf_counter_ns()
                for _ in range(repetitions):
                    operation()
                timings[name].append((perf_counter_ns() - start) / repetitions)
        results.append({
            "address_space_bytes": size,
            "median_ns": {name: statistics.median(raw) for name, raw in timings.items()},
            "raw_ns_per_operation": timings,
        })
    print(json.dumps({"torch": torch.__version__, "threads": torch.get_num_threads(),
                      "samples": samples, "operations_per_sample": repetitions,
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
