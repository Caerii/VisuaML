"""Build the two fixed ordinary-transition graphs used in the experiment."""
import json
from pathlib import Path
from time import perf_counter_ns

from visuaml.compiler_lab.frozen_ordinary_v58 import freeze_ordinary

OUT = Path(__file__).resolve().parents[1] / "evidence" / "ordinary-effects"


def main():
    costs = {}
    for name, lazy in (("descriptor", False), ("lazy", True)):
        started = perf_counter_ns()
        graph = freeze_ordinary(lazy_fields=lazy)
        costs[name] = perf_counter_ns() - started
        graph.save(str(OUT / f"{name}.pt"))
        ir = str(graph.inlined_graph)
        assert "prim::Loop" in ir and "prim::PythonOp" not in ir
        (OUT / f"{name}.ir").write_text(ir)
    print(json.dumps({"freeze_ns": costs}, indent=2))


if __name__ == "__main__":
    main()
