"""V8: holdout-measured evidence into per-node mincut cost, fail closed."""
from __future__ import annotations

from dataclasses import replace

import pytest

from visuaml.compiler_lab.backends import BackendCapabilities
from visuaml.compiler_lab.calibration_evidence import validate_latency_holdout
from visuaml.compiler_lab.evidence_placement import optimize_with_evidence
from visuaml.compiler_lab.hardware_calibration import CalibrationError, LatencyObservation
from visuaml.compiler_lab.placement_optimizer import HypotheticalCostModel, optimize_binary_placement
from visuaml.compiler_spike import OpRecord, ProgramIR, TensorSpec


def evidence(hardware, op, coefficients, source="synthetic"):
    a,b,c=coefficients
    records=[]
    for i in range(32):
        w = 1 + (i * 47) % 193
        size = 16 + (i * 37) % 199
        records.append(LatencyObservation(
            measurement_id=f"{hardware}-{op}-{i:02d}", hardware_id=hardware,
            operation=op, source_kind=source, workload_units=w, bytes_moved=size,
            latency_us=a + b*w + c*size, provenance="v8-synthetic-evidence-fixture"))
    train=records[:24]
    holdout=records[24:]
    return validate_latency_holdout(train,holdout,allowed_max_abs_us=1e-7,allowed_max_relative=1e-7)


def sample():
    f=lambda n: (TensorSpec(("1","4"), "torch.float32", n),)
    ir=ProgramIR((
        OpRecord("x","input","",(),f(32),0,()),
        OpRecord("m1","matmul","",("x",),f(48),80,()),
        OpRecord("r","relu","",("m1",),f(48),12,()),
        OpRecord("m2","matmul","",("r",),f(64),130,()),
        OpRecord("o","output","",("m2",),(),0,()),
    ),("x",),("m2",))
    caps=BackendCapabilities("demo-device","revision-1",frozenset({"matmul"}),frozenset({"relu"}),frozenset({"torch.float32"}))
    host={"matmul":evidence("cpu","matmul",(6,0.3,0.005)), "relu":evidence("cpu","relu",(1,0.04,0.001))}
    device={"matmul":evidence("optical-surrogate","matmul",(0.2,0.01,0.001))}
    transfer=evidence("host-link","transfer",(0.02,0.1,0.006))
    return ir,caps,host,device,transfer


def run(allow=True):
    ir,caps,host,device,transfer=sample()
    return optimize_with_evidence(ir,caps,host_hardware_id="cpu",device_hardware_id="optical-surrogate",transfer_hardware_id="host-link",host=host,device=device,transfer=transfer,allow_synthetic=allow)


def test_per_node_estimates_differ_for_same_operator():
    report=run()
    costs=dict(report.predicted_device_us)
    assert costs["m2"]>costs["m1"]
    assert report.placement.estimated_total_cost <= report.placement.all_host_baseline
    assert report.unit=="microseconds" and report.source_kind=="synthetic"
    assert report.evidence_digests and len(report.input_sha256)==64
    assert "measurement" in " ".join(report.warnings)


def test_evidence_matches_direct_per_node_graph_cut_objective():
    ir,caps,host,device,transfer=sample()
    result=run()
    host_costs=dict(result.predicted_host_us)
    device_costs=dict(result.predicted_device_us)
    profile=HypotheticalCostModel(host_node_costs=host_costs,device_node_costs=device_costs,
        transfer_overhead=result.transfer_overhead_us,transfer_bytes_per_unit=1/result.transfer_us_per_byte)
    reference=optimize_binary_placement(ir,caps,profile)
    assert result.placement.estimated_total_cost==pytest.approx(reference.estimated_total_cost)
    assert [x.assigned for x in result.placement.assignments]==[x.assigned for x in reference.assignments]


def test_synthetic_data_cannot_be_silently_called_measured():
    with pytest.raises(CalibrationError,match="Synthetic"):
        run(allow=False)


def test_reject_bad_evidence_coverage_or_device_identity():
    ir,caps,host,device,transfer=sample()
    with pytest.raises(CalibrationError,match="coverage"):
        optimize_with_evidence(ir,caps,host_hardware_id="cpu",device_hardware_id="optical-surrogate",transfer_hardware_id="host-link",host={"matmul":host["matmul"]},device=device,transfer=transfer,allow_synthetic=True)
    with pytest.raises(CalibrationError,match="Device evidence"):
        optimize_with_evidence(ir,caps,host_hardware_id="cpu",device_hardware_id="other",transfer_hardware_id="host-link",host=host,device=device,transfer=transfer,allow_synthetic=True)


def test_reject_failed_holdout_and_extrapolation():
    ir,caps,host,device,transfer=sample()
    bad=dict(host)
    bad["relu"]=replace(bad["relu"],passed=False)
    with pytest.raises(CalibrationError,match="Failed holdout"):
        optimize_with_evidence(ir,caps,host_hardware_id="cpu",device_hardware_id="optical-surrogate",transfer_hardware_id="host-link",host=bad,device=device,transfer=transfer,allow_synthetic=True)
    nodes=list(ir.nodes)
    nodes[1]=replace(nodes[1],estimated_flops=5000)
    with pytest.raises(CalibrationError,match="outside"):
        optimize_with_evidence(replace(ir,nodes=tuple(nodes)),caps,host_hardware_id="cpu",device_hardware_id="optical-surrogate",transfer_hardware_id="host-link",host=host,device=device,transfer=transfer,allow_synthetic=True)


def test_dataset_change_changes_report_fingerprint():
    ir,caps,host,device,transfer=sample()
    report=run()
    other=evidence("host-link","transfer",(0.02,0.1,0.008))
    changed=optimize_with_evidence(ir,caps,host_hardware_id="cpu",device_hardware_id="optical-surrogate",transfer_hardware_id="host-link",host=host,device=device,transfer=other,allow_synthetic=True)
    assert report.input_sha256 != changed.input_sha256


def test_independent_proof_of_binary_optimum_on_small_graph():
    import itertools
    ir,caps,host,device,transfer=sample()
    r=run()
    cpu=dict(r.predicted_host_us)
    acc=dict(r.predicted_device_us)
    widths={n.id:sum(t.bytes or 0 for t in n.tensors) for n in ir.nodes}
    values=[]
    for a,b in itertools.product((False,True),repeat=2):
        loc={"x":"host","m1":"device" if a else "host","r":"host","m2":"device" if b else "host","o":"host"}
        total=sum((acc[n] if loc[n]=="device" else cpu[n]) for n in cpu)
        for node in ir.nodes:
            for dep in node.inputs:
                if loc[node.id]!=loc[dep]:
                    total+=r.transfer_overhead_us + widths[dep]*r.transfer_us_per_byte
        values.append(total)
    assert min(values)==pytest.approx(r.placement.estimated_total_cost)
