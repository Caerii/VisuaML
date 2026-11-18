from __future__ import annotations

import json
import subprocess
from functools import partial
from pathlib import Path
from time import perf_counter_ns
import sys

from scripts.benchmark_neural_vm_v58 import summarize_ns
from visuaml.compiler_lab.neural_microstep_executor_v58 import execute_microstep_control
from visuaml.compiler_lab.wasm_research_corpus_v57 import generated

program = generated(7)
pairs = {}
for traced in (True, False):
    fn = {
        name: partial(execute_microstep_control, program, -1, fuel=1500,
                      capture_trace=traced, use_compiled_alu=(name == "compiled"))
        for name in ("numpy", "compiled")
    }
    # Warm both modes and check full result identity.
    expected = fn["numpy"]()
    assert fn["compiled"]() == expected
    samples = {name: [] for name in fn}
    for trial in range(15):
        names = tuple(fn) if trial % 2 == 0 else tuple(reversed(fn))
        for name in names:
            start = perf_counter_ns()
            actual = fn[name]()
            samples[name].append(perf_counter_ns() - start)
            assert actual == expected
    pairs["trace" if traced else "no_trace"] = {
        name: summarize_ns(times) for name, times in samples.items()
    }
commit, tree = subprocess.check_output(("git", "rev-parse", "HEAD", "HEAD^{tree}"), text=True).splitlines()
data = {"source": {"commit": commit, "tree": tree}, "program": "generated_07", "input": -1,
        "purpose": "Focused reproduction of the largest same-mode no-trace regression",
        "repeats": 15, "paired": pairs}
with Path(sys.argv[1]).open("x") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
