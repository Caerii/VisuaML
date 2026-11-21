"""Inspect admitted program artifacts and compare aligned exported tensor graphs."""
from pathlib import Path

import torch
import wasmtime
from torch.fx import Interpreter

from visuaml.compiler_lab.frozen_calls_v58 import load_call_image
from visuaml.compiler_lab.wasm_bitstream_v39 import _read_u32, parse_section_envelope


def describe_wasm(raw):
    try:
        wasmtime.Module.validate(wasmtime.Engine(), raw)
    except wasmtime.WasmtimeError as error:
        return dict(format="wasm", valid=False, replayable=False, reason=str(error))
    report = dict(format="wasm", valid=True, replayable=False)
    try:
        sections = parse_section_envelope(raw)
        report["core_sections"] = [section.section_id for section in sections if section.section_id]
        report["custom_sections"] = []
        for section in sections:
            if section.section_id == 0:
                size, start = _read_u32(section.payload, 0)
                report["custom_sections"].append(section.payload[start:start+size].decode("utf-8"))
        image = load_call_image(raw)
    except ValueError as error:
        return dict(report, reason=str(error))
    return dict(report, replayable=True, memory_pages=image.memory_pages,
                active_data_bytes=sum(len(data) for _, data in image.data_segments), functions=image.functions)


def describe_onnx(raw):
    """Read ONNX metadata without loading external weights or running operators."""
    if len(raw) > 16*1024*1024:
        raise ValueError("ONNX inspection limit is 16 MiB")
    try:
        import onnx
        from google.protobuf.message import DecodeError
    except ImportError as error:
        raise ValueError("ONNX inspection requires the visuaml-replay[onnx] extra") from error
    try:
        model = onnx.load_model_from_string(raw)
    except DecodeError as error:
        raise ValueError("Invalid ONNX protobuf") from error
    if not model.HasField("graph"):
        raise ValueError("ONNX model has no graph")

    def bindings(values):
        result = []
        for value in values:
            item = dict(name=value.name)
            if value.type.HasField("tensor_type"):
                tensor = value.type.tensor_type
                item["dtype"] = tensor.elem_type
                if tensor.HasField("shape"):
                    item["shape"] = [d.dim_value if d.HasField("dim_value") else d.dim_param or None
                                     for d in tensor.shape.dim]
            result.append(item)
        return result

    graph = model.graph
    return dict(format="onnx", inspectable=True, execution=None, validated=False,
                opsets={item.domain: item.version for item in model.opset_import},
                inputs=bindings(graph.input), outputs=bindings(graph.output),
                initializers=[dict(name=item.name, external=item.data_location == onnx.TensorProto.EXTERNAL)
                              for item in graph.initializer],
                nodes=[dict(name=node.name, operator=node.op_type, domain=node.domain,
                            inputs=list(node.input), outputs=list(node.output),
                            has_subgraphs=any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                                              for attr in node.attribute))
                       for node in graph.node])


def describe_export(exported):
    """Inspect a trusted torch.export.ExportedProgram without lowering its operators."""
    nodes = []
    for node in exported.graph.nodes:
        value = node.meta.get("val")
        row = dict(name=node.name, kind=node.op, operator=str(node.target),
                   inputs=[arg.name for arg in node.all_input_nodes])
        if isinstance(value, torch.Tensor):
            row.update(shape=[str(dim) for dim in value.shape], dtype=str(value.dtype))
        nodes.append(row)
    return dict(format="torch.export", inspectable=True, execution="PyTorch",
                input_bindings=[dict(kind=spec.kind.name, target=spec.target, name=spec.arg.name)
                                for spec in exported.graph_signature.input_specs],
                range_constraints={str(key): str(value) for key, value in exported.range_constraints.items()},
                nodes=nodes)


class _TensorTrace(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.observed = {}

    def run_node(self, node):
        value = super().run_node(node)
        # Retain a snapshot since a later operation may mutate an alias.
        if isinstance(value, torch.Tensor):
            self.observed[node.name] = value.detach().clone()
        return value


def _signature(graph):
    return [(node.name, node.op, str(node.target), repr(node.args), repr(node.kwargs)) for node in graph.nodes]


def _difference(expected, actual, rtol, atol):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return dict(reason="shape-or-dtype", expected_shape=list(expected.shape), actual_shape=list(actual.shape),
                    expected_dtype=str(expected.dtype), actual_dtype=str(actual.dtype))
    same = (torch.isclose(actual, expected, rtol=rtol, atol=atol)
            if expected.is_floating_point() or expected.is_complex() else expected == actual)
    if bool(same.all()):
        return None
    index = tuple((~same).nonzero()[0].tolist())
    return dict(reason="value", index=list(index), expected=str(expected[index].item()),
                actual=str(actual[index].item()), dtype=str(expected.dtype), shape=list(expected.shape))


def compare_exports(reference, candidate, inputs, *, rtol=0.0, atol=0.0):
    """Compare aligned single-tensor stages in trusted exports on flat inputs."""
    if any(module.graph_signature.buffers_to_mutate or module.graph_signature.user_inputs_to_mutate
           for module in (reference, candidate)):
        raise ValueError("Activation comparison requires exports without in-place state or input updates")
    if not 0 <= rtol < float("inf") or not 0 <= atol < float("inf"):
        raise ValueError("Tolerances must be finite and nonnegative")
    left, right = reference.module(), candidate.module()
    if _signature(left.graph) != _signature(right.graph):
        raise ValueError("Activation comparison requires aligned exported graph structure")
    traces = [_TensorTrace(module) for module in (left, right)]
    with torch.inference_mode():
        for module, trace in zip((left, right), traces):
            cloned = tuple(arg.detach().clone() for arg in inputs)
            result = module(*cloned)
            if not isinstance(result, torch.Tensor):
                raise ValueError("Activation comparison requires a single tensor output")
            observed = trace.run(*cloned)
            if not isinstance(observed, torch.Tensor) or _difference(result, observed, 0, 0) is not None:
                raise ValueError("Activation tracing did not reproduce the deterministic export output")
    first = activation = None
    compared = 0
    for node in left.graph.nodes:
        if node.name not in traces[0].observed:
            continue
        compared += 1
        if node.name not in traces[1].observed:
            raise ValueError(f"Tensor-valued node differs at {node.name}")
        delta = _difference(traces[0].observed[node.name], traces[1].observed[node.name], rtol, atol)
        if delta is not None:
            mismatch = dict(node=node.name, operator=str(node.target), **delta)
            if first is None:
                first = mismatch
            if activation is None and node.op.startswith("call_"):
                activation = mismatch
    return dict(format="torch.export", execution="PyTorch", compared_tensor_nodes=compared,
                rtol=rtol, atol=atol, first_difference=first, first_activation_difference=activation)


def load_export(path: Path):
    """Load a PyTorch export from a trusted sender."""
    return torch.export.load(path)
