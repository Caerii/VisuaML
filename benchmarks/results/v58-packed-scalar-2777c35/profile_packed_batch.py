"""Sample the untraced serial batch kernel with an instrumented C++ driver."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.benchmark_native_batch_loop_v58 import CONTROL
from scripts.benchmark_neural_vm_v58 import git_identity, workloads
from visuaml.compiler_lab.neural_control_routing_v52 import TERMINAL
from visuaml.compiler_lab.neural_microstep_executor_v58 import PreparedMicrostepProgram

root = Path.cwd()
source = git_identity(root)
kernel = root / "visuaml-client/backend/visuaml/compiler_lab/constructed_alu_v58.cpp"
corpus = sorted(
    (work for work in workloads(scaling=True)
     if not any(op.opcode in CONTROL for op in work.program.operations[:-1])),
    key=lambda work: (len(work.program.operations), work.name),
)
work = corpus[len(corpus) // 2]
program = work.program
prepared = PreparedMicrostepProgram(program, use_native_loop=True, use_packed_scalar=True)
native = prepared._inputs.native_microstep
driver = [
    '#include <cstddef>',
    '#include <cstdint>',
    '#include <iostream>',
    '''extern "C" void visuaml_prepared_batch(
        const std::int32_t*, const std::int32_t*, std::size_t, std::size_t, std::size_t,
        std::size_t, std::size_t, const double*, const double*, const std::int32_t*,
        const double*, const double*, const double*, const double*, std::int32_t*,
        std::int32_t*, std::int32_t*, double*, bool);''',
]


def declare(name, values, c_type):
    data = ", ".join(map(str, values))
    driver.append(f"static const {c_type} {name}[] = {{{data}}};")


declare("rows", native._rows.tolist(), "double")
declare("fixed", native._fixed.tolist(), "double")
names = ("targets", "guards", "guard_weights", "path_weights", "and_weight")
for name, tensor in zip(names, native._router._tensors, strict=True):
    declare(name, tensor.ravel().tolist(), "std::int32_t" if name == "targets" else "double")
size, repetitions = 1024, 2400
arguments = tuple(work.arguments[i % len(work.arguments)] for i in range(size))
declare("arguments", arguments, "std::int32_t")
declare("fuels", [400] * size, "std::int32_t")
driver.append(
    f'''
int main() {{
    std::int32_t results[{2 * size}], statuses[{size}];
    std::int64_t checksum = 0;
    for (int repetition = 0; repetition < {repetitions}; ++repetition) {{
        visuaml_prepared_batch(arguments, fuels, {size}, {len(program.operations)}, {TERMINAL},
            {program.local_count + 1}, 0, rows, fixed, {", ".join(names)},
            results, statuses, nullptr, nullptr, true);
        for (int lane = 0; lane < {size}; ++lane) {{
            if (statuses[lane]) return 1;
            checksum += results[lane * 2];
        }}
    }}
    std::cout << checksum << '\\n';
}}
'''
)
flags = ["-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off", "-pg"]
with tempfile.TemporaryDirectory(prefix="v58-batch-profile-") as tmp:
    path = Path(tmp)
    (path / "driver.cpp").write_text("\n".join(driver))
    subprocess.run(["c++", *flags, kernel, "driver.cpp", "-o", "run"], check=True, cwd=path)
    checksum = int(subprocess.check_output(["./run"], cwd=path))
    expected = sum(result.result_i32 for result in prepared.run_many(arguments, capture_trace=False))
    assert checksum == repetitions * expected
    profile = subprocess.check_output(["gprof", "-b", "run", "gmon.out"], text=True, cwd=path)
compiler = subprocess.check_output(["c++", "--version"], text=True).splitlines()[0]
with Path(sys.argv[1]).open("x") as stream:
    json.dump({"source": source, "kernel_source_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
               "compiler": compiler, "flags": flags, "program": work.name, "batch_size": size,
               "repetitions": repetitions, "checksum": checksum, "gprof": profile},
              stream, sort_keys=True, indent=2)
    stream.write("\n")
