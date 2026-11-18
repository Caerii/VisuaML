"""Check the emitted AVX2 arithmetic in the same kernel used by the benchmark."""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.benchmark_neural_vm_v58 import git_identity
from visuaml.compiler_lab.constructed_alu_v58 import compiled_alu

root = Path.cwd()
source = git_identity(root)
kernel = compiled_alu()
assert kernel.prepared_simd_available()
cpp = root / "visuaml-client/backend/visuaml/compiler_lab/constructed_alu_v58.cpp"
with tempfile.TemporaryDirectory() as directory:
    binary = Path(directory) / "alu.so"
    subprocess.run(["c++", *kernel.flags, cpp, "-o", binary], check=True)
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == kernel.binary_sha256
    disassembly = subprocess.check_output(["objdump", "-d", "-C", binary], text=True)

symbol = re.search(r"^[0-9a-f]+ <\(anonymous namespace\)::prepared_avx2\([^\n]+:\n(.*?)(?=\n\n)",
                   disassembly, re.MULTILINE | re.DOTALL)
assert symbol
evidence = [line.strip() for line in symbol[1].splitlines()
            if "%ymm" in line and re.search(r"\b(?:vpaddd|vpsubd|vpmulld|vpcmpeqd|vpcmpgtd)\b", line)]
for op in ("vpaddd", "vpsubd", "vpmulld", "vpcmpeqd"):
    assert any(re.search(rf"\b{op}\b", line) for line in evidence), op
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "kernel_sha256": kernel.binary_sha256, "compiler": kernel.compiler,
               "avx2_available": True, "instructions": evidence}, stream, indent=2, sort_keys=True)
    stream.write("\n")
