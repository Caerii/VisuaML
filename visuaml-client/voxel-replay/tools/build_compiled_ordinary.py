"""Build and register the one CPU graph operator, then freeze its caller."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from visuaml.compiler_lab import frozen_ordinary_v58
from visuaml.compiler_lab.ordinary_native_v58 import register

OUT = Path(__file__).resolve().parents[1] / "evidence" / "compiled-ordinary-cpu"


def main():
    argparse.ArgumentParser().parse_args()
    data = register()
    started = perf_counter_ns()
    machine = frozen_ordinary_v58.freeze_ordinary(lazy_fields=True, native_runs=True)
    data["freeze_ns"] = perf_counter_ns() - started
    machine.save(str(OUT / "machine.pt"))
    ir = str(machine.inlined_graph)
    assert "visuaml_ordinary::run" in ir and "prim::Loop" in ir and "prim::PythonOp" not in ir
    (OUT / "machine.ir").write_text(ir)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
