"""Compare legacy token identities against the archived indexed-gate corpus."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import git_identity, workloads
from visuaml.compiler_lab.neural_control_routing_v52 import compile_route_tokens
from visuaml.compiler_lab.typed_microop_ir_v57 import compile_microop_tokens
from visuaml.compiler_lab.wasm_control_flow_v51 import encode_control

root = Path.cwd()
source = git_identity(root)
path = root / "benchmarks/results/v58-indexed-gates-7160363/runtime-1.json.gz"
reference = json.loads(gzip.decompress(path.read_bytes()))
programs = {work.name: work.program for work in workloads(scaling=True)}
for row in reference["rows"]:
    program = programs[row["program"]]
    assert hashlib.sha256(encode_control(program)).hexdigest() == row["source_wasm_sha256"]
    assert compile_route_tokens(program).digest() == row["route_tokens_sha256"]
    assert compile_microop_tokens(program).digest() == row["microop_tokens_sha256"]
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "reference": str(path.relative_to(root)), "programs": len(reference["rows"])},
              stream, indent=2, sort_keys=True)
    stream.write("\n")
