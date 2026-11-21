"""Versioned, data-only checkpoint for a running frozen WASM direct-call state."""
import base64
from time import perf_counter_ns

import torch

from visuaml.compiler_lab.frozen_calls_v58 import CallState, initialize
from visuaml.compiler_lab.frozen_recurrent_v58 import MachineState


from ._codec import canonical, digest


def identities(graph_bytes, image):
    return {"graph_sha256": digest(graph_bytes), "wasm_sha256": image.binary_sha256,
            "image_sha256": image.image_sha256}


def encode(state, identity):
    fields = {name: value.tolist() for name, value in zip(state.vm._fields, state.vm) if name != "page"}
    payload = dict(version=1, identities=identity, registers=fields,
                   depth=int(state.depth), frames=state.frames.tolist(),
                   page=base64.b64encode(state.vm.page.numpy().tobytes()).decode("ascii"))
    return dict(payload, checkpoint_sha256=digest(canonical(payload)))


def restore(doc, identity, image, profile=None):
    started = perf_counter_ns()
    payload = {key: value for key, value in doc.items() if key != "checkpoint_sha256"}
    if payload.get("version") != 1 or payload.get("identities") != identity:
        raise ValueError("Checkpoint version or execution identity mismatch")
    if digest(canonical(payload)) != doc.get("checkpoint_sha256"):
        raise ValueError("Checkpoint integrity mismatch")
    checked = perf_counter_ns()
    length = image.memory_pages * 65536
    page = base64.b64decode(payload["page"], validate=True)
    if len(page) != length:
        raise ValueError("Checkpoint page length mismatch")
    decoded = perf_counter_ns()
    prototype = initialize(image, 0, page, 0)

    allocated = perf_counter_ns()

    def tensor(value, expected):
        actual = torch.as_tensor(value)
        if actual.dtype != torch.int64 or actual.shape != expected.shape:
            raise ValueError("Checkpoint register shape or integer type mismatch")
        return actual

    regs = payload["registers"]
    values = [prototype.vm.page if key == "page" else tensor(regs[key], getattr(prototype.vm, key))
              for key in MachineState._fields]
    vm = MachineState(*values)
    depth = tensor(payload["depth"], prototype.depth)
    frames = tensor(payload["frames"], prototype.frames)
    end = len(image.rows) - 1
    if (not (0 <= int(vm.pc) < end and 0 <= int(vm.height) <= vm.stack.numel()
             and int(vm.fuel) >= 0 and int(vm.status) == 0 and 0 <= int(depth) <= len(frames))):
        raise ValueError("Checkpoint is not a valid live continuation")
    for pc, height in frames[:int(depth), :2].tolist():
        if not (0 <= pc < end and 1 <= height <= vm.stack.numel()):
            raise ValueError("Checkpoint caller frame is invalid")
    if profile is not None:
        profile.update(integrity_ns=checked-started, page_decode_ns=decoded-checked,
                       page_initialize_ns=allocated-decoded, registers_ns=perf_counter_ns()-allocated)
    return CallState(vm, depth, frames)
