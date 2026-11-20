"""Build and register the one CPU graph operator, then freeze its caller."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns

import torch

from visuaml.compiler_lab import frozen_ordinary_v58

OUT = Path(__file__).resolve().parents[1] / "evidence" / "compiled-ordinary-cpu"
SOURCE = Path(frozen_ordinary_v58.__file__).with_name("native") / "ordinary_cpu_v58.cpp"


def register():
    compiler = os.environ.get("CXX", "c++")
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib = torch_root / "lib"
    source_hash = sha256(SOURCE.read_bytes()).hexdigest()
    abi = int(torch.compiled_with_cxx11_abi())
    key = f"{source_hash}:{version}:{torch.__version__}:{torch_root}:{abi}"
    cache = Path("/tmp") / ("visuaml-ordinary-" + sha256(key.encode()).hexdigest()[:16])
    cache.mkdir(exist_ok=True)
    artifact = cache / "ordinary.so"
    built = not artifact.exists()
    started = perf_counter_ns()
    if built:
        pending = cache / f"ordinary-{os.getpid()}.so"
        subprocess.run([
            compiler, "-shared", "-fPIC", "-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            f"-I{torch_root / 'include'}", str(SOURCE), f"-L{torch_lib}", f"-Wl,-rpath,{torch_lib}",
            "-ltorch_cpu", "-ltorch", "-lc10", "-o", str(pending),
        ], check=True)
        pending.replace(artifact)
    build_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    torch.ops.load_library(str(artifact))
    return {"compiled": built, "build_ns": build_ns, "register_ns": perf_counter_ns() - started,
            "source_sha256": source_hash, "library_sha256": sha256(artifact.read_bytes()).hexdigest(),
            "compiler": version, "torch": torch.__version__}


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
