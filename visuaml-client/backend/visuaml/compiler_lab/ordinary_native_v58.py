"""Build and register the repository's ordinary CPU graph operator."""
from hashlib import sha256
import os
from pathlib import Path
from stat import S_ISDIR
import subprocess
from tempfile import TemporaryDirectory, gettempdir
from time import perf_counter_ns

import torch

SOURCE = Path(__file__).with_name("native") / "ordinary_cpu_v58.cpp"


def register():
    compiler = os.environ.get("CXX", "c++")
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib = torch_root / "lib"
    source_hash = sha256(SOURCE.read_bytes()).hexdigest()
    abi = int(torch.compiled_with_cxx11_abi())
    key = f"{source_hash}:{version}:{torch.__version__}:{torch_root}:{abi}"
    cache = Path(gettempdir()) / f"visuaml-ordinary-{os.getuid()}-{sha256(key.encode()).hexdigest()[:16]}"
    cache.mkdir(mode=0o700, exist_ok=True)
    info = cache.lstat()
    if not S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("Compiled runner cache must be a private directory")
    artifact = cache / "ordinary.so"
    built = not artifact.exists()
    started = perf_counter_ns()
    if built:
        with TemporaryDirectory(dir=cache) as staging:
            pending = Path(staging) / "ordinary.so"
            subprocess.run([
                compiler, "-shared", "-fPIC", "-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
                f"-I{torch_root / 'include'}", str(SOURCE), f"-L{torch_lib}", f"-Wl,-rpath,{torch_lib}",
                "-ltorch_cpu", "-ltorch", "-lc10", "-o", str(pending),
            ], check=True)
            pending.replace(artifact)
    build_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    if str(artifact.resolve()) not in torch.ops.loaded_libraries:
        torch.ops.load_library(str(artifact))
    return {"compiled": built, "build_ns": build_ns, "register_ns": perf_counter_ns() - started,
            "source_sha256": source_hash, "library_sha256": sha256(artifact.read_bytes()).hexdigest(),
            "compiler": version, "torch": torch.__version__}
