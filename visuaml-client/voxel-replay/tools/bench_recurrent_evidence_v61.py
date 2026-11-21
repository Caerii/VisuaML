"""Compare persistent validator placements and replay a typed tensor-VM query."""
from argparse import ArgumentParser
import ctypes
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import torch

from evaluate_persistent_evidence_v60 import programs, timed
from frozen_call_checkpoint import canonical, digest
from persistent_evidence_v60 import Node, Store, affine, identity, prepare, publish, smt, validate
from recurrent_evidence_v61 import OUT
from replay_frozen_recurrent import PACKED, wasmtime_run
from replay_guest_rewrite import EXECUTOR, invoke


def bench(directory):
    store = Store(Path(directory)/"dag")
    _,source,candidate,_ = list(programs())[3]
    scope = prepare(source,candidate)
    (root,_),proof_publish = timed(lambda: publish(store,scope))
    code = Path(__file__).with_name("affine_validate_v61.c")
    library = Path(directory)/"affine.so"
    subprocess.run(["cc","-O2","-shared","-fPIC","-o",str(library),str(code)],check=True)
    function = ctypes.CDLL(str(library)).affine_v61
    function.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint32)]
    function.restype = ctypes.c_int
    vocabulary = {name:i for i,name in enumerate(("x","const","add","sub","mul"))}

    def native(tokens):
        flat = [word for op,imm in tokens for word in (vocabulary[op],imm)]
        values = (ctypes.c_int32*len(flat))(*flat)
        out = (ctypes.c_uint32*2)()
        if function(values,len(tokens),out)!=1:
            raise ValueError("native validator rejected the expression")
        return list(out)

    for _,original,alternative,_ in programs():
        for instructions in (original,alternative):
            assert native(instructions)==affine(instructions)
    worker = Path(__file__).with_name("persistent_evidence_v60.py")
    launched = perf_counter_ns()
    process = subprocess.Popen([sys.executable,str(worker),str(store.root)],stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,text=True)
    assert process.stdout.readline().strip()=="ready"
    startup = perf_counter_ns()-launched
    times = dict(python=[],native=[],service=[],worker=[],startup_ns=startup,proof_publish_ns=proof_publish)
    try:
        for _ in range(24):
            ok,elapsed = timed(lambda: validate(store,root,prepare(source,candidate)))
            assert ok
            times["python"].append(elapsed)
            # Scope construction stays identical; only the normalizer for
            # the independently rechecked witness is replaced.
            repeated_scope,scope_ns = timed(lambda: prepare(source,candidate))
            with patch("persistent_evidence_v60.affine",native):
                ok,elapsed = timed(lambda: validate(store,root,repeated_scope))
            assert ok
            times["native"].append(scope_ns+elapsed)
            started = perf_counter_ns()
            process.stdin.write(json.dumps(dict(source=source,candidate=candidate,root=root))+"\n")
            process.stdin.flush()
            report = json.loads(process.stdout.readline())
            assert report["accepted"]
            times["service"].append(perf_counter_ns()-started)
            times["worker"].append(report)
    finally:
        process.stdin.close()
        assert process.wait(timeout=15)==0
    _,times["invalidation_ns"] = timed(lambda: store.affected([identity(scope[1][0])]))
    return times


def tensor_sample():
    # Two programs agree at zero, yet differ for other i32 inputs.
    source = [("x",0)]
    alternative = [("x",0),("const",2),("mul",0)]
    scope = prepare(source,alternative)
    machine = EXECUTOR/"machine.pt"
    graph = torch.jit.load(str(machine))
    page = bytearray(65536)
    query = Node("query",dict(module="exact-cpu-tensor-vm",authority="sample",
                             argument=0,fuel=200,graph=digest(machine.read_bytes()),
                             page=digest(page)),tuple(identity(n) for n in scope[1]))
    results,timings = [],[]
    with patch(PACKED,side_effect=AssertionError("packed fallback")):
        for binary in scope[2]:
            (output,result),execution = timed(lambda: invoke(graph,binary,0,page))
            native,baseline = timed(lambda: wasmtime_run(binary,0,page))
            assert native==(result["return_i32"],output)
            results.append(result["return_i32"])
            timings.append(dict(frozen_ns=execution,native_ns=baseline))
    assert results==[0,0]
    valid,witness = smt(source,alternative)
    assert not valid
    return dict(query=json.loads(query.bytes()),witness=dict(results=results,strong_authorized=False,
              counterexample=witness)),timings


def measured_payback():
    _,source,candidate,_ = list(programs())[3]
    scope = prepare(source,candidate)
    graph = torch.jit.load(str(EXECUTOR/"machine.pt"))
    (ok,_),verification = timed(lambda: smt(source,candidate))
    assert ok
    elapsed = [[],[]]
    with patch(PACKED,side_effect=AssertionError("packed fallback")):
        for argument in range(13):
            for i,binary in enumerate(scope[2]):
                (page,result),duration = timed(lambda: invoke(graph,binary,argument,bytearray(65536)))
                assert result["return_i32"]==argument and page==bytes(65536)
                if argument:
                    elapsed[i].append(duration)
    return dict(source_ns=elapsed[0],candidate_ns=elapsed[1],verification_ns=verification,
                net_ns=sum(elapsed[0])-sum(elapsed[1])-verification)


def main():
    parser = ArgumentParser()
    parser.add_argument("--write",action="store_true")
    opts = parser.parse_args()
    torch.set_num_threads(1)
    with TemporaryDirectory() as folder:
        timing = bench(folder)
    result,execution = tensor_sample()
    if opts.write:
        (OUT/"module.json").write_bytes(canonical(result)+b"\n")
        (OUT/"validation-timings.json").write_bytes(canonical(dict(validation=timing,execution=execution,payback=measured_payback()))+b"\n")
    else:
        assert result==json.loads((OUT/"module.json").read_bytes())


if __name__=="__main__":
    main()
