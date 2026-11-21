"""Pinned source-instruction reference runtime for the direct-call profile."""
from io import BytesIO
from pathlib import Path

import torch

from visuaml.compiler_lab.frozen_calls_v58 import CallState, initialize, load_call_image
from .compiled import MAX_BATCH, CompiledRunner
from .checkpoint import digest, encode, identities, restore

GRAPH = Path(__file__).with_name("assets") / "reference.pt"
PAGE_BYTES = 65536


class Replay:
    def __init__(self, raw: bytes, graph_path: Path = GRAPH):
        self.raw = raw
        self.image = load_call_image(raw)
        self._reference_graph_path = graph_path
        graph_bytes = graph_path.read_bytes()
        self.graph = torch.jit.load(BytesIO(graph_bytes), map_location="cpu")
        self.identity = identities(graph_bytes, self.image)

    def run(self, state: CallState, steps: int | None = None, *, runner: str = "reference") -> CallState:
        """Advance up to a source-step limit, or stop at halt, trap, or fuel exhaustion."""
        if steps is not None and (type(steps) is not int or steps < 0):
            raise ValueError("Source-step limit must be a nonnegative integer")
        if runner not in ("reference", "compiled"):
            raise ValueError("Unknown runner")
        if runner == "compiled" and steps != 0 and int(state.vm.status) == 0:
            if not hasattr(self, "_compiled"):
                if self._reference_graph_path != GRAPH and self.identity["graph_sha256"] != digest(GRAPH.read_bytes()):
                    raise ValueError("Compiled runner requires the release reference graph")
                self._compiled = CompiledRunner(self.image.rows)
            while steps != 0 and int(state.vm.status) == 0:
                budget = MAX_BATCH if steps is None else min(MAX_BATCH, steps)
                state = self._compiled.advance(state, budget)
                if steps is not None:
                    steps -= budget
        else:
            while steps != 0 and int(state.vm.status) == 0:
                state = self.advance(state)
                if steps is not None:
                    steps -= 1
        return state

    def summary(self, state: CallState) -> dict:
        vm = state.vm
        return dict(status=int(vm.status), fuel=int(vm.fuel),
                    returned=int(vm.stack[0]) if int(vm.status) == 1 else None,
                    page_sha256=digest(vm.page.numpy().tobytes()))

    def start(self, argument: int, page: bytes, fuel: int) -> CallState:
        """Allocate a private page and begin a new invocation."""
        return initialize(self.image, argument, page, fuel)

    def restore(self, checkpoint: dict) -> CallState:
        """Validate identity and restore an independent mutable state."""
        return restore(checkpoint, self.identity, self.image)

    def checkpoint(self, state: CallState) -> dict:
        """Publish an independent continuation at a running source boundary."""
        if int(state.vm.status) != 0:
            raise ValueError("Checkpoint requires a running state")
        return encode(state, self.identity)

    def advance(self, state: CallState) -> CallState:
        """Advance one source transition without serializing an observation."""
        if int(state.vm.status) != 0:
            raise ValueError("Cannot advance a stopped run")
        return self.graph(self.image.rows, state).state

    def step(self, state, preview=None):
        pc = int(state.vm.pc)
        function = max(i for i, start in enumerate(self.image.starts) if start <= pc)
        source = pc-self.image.starts[function]
        opcode = self.image.functions[function][source]
        pulse = self.graph(self.image.rows, state)
        vm = pulse.state.vm
        display = pulse.state
        if preview is not None:
            if self.image.functions[function][source:source+2] != ("local.get", "local.get"):
                raise ValueError("Preview compare requires a local.get pair at the checkpoint")
            if not 0 <= preview < state.vm.locals.numel() or int(vm.status) != 0:
                raise ValueError("Preview local or source boundary is invalid")
            stack = vm.stack.clone()
            stack[int(vm.height)-1] = vm.locals[preview]
            display = display._replace(vm=vm._replace(stack=stack))
        row = dict(pc=pc, function=function, source_index=source, opcode=opcode,
                   next_pc=int(vm.pc), fuel=int(vm.fuel), status=int(vm.status),
                   depth=int(display.depth), stack=display.vm.stack[:int(vm.height)].tolist(),
                   access=pulse.access.tolist(), state_sha256=encode(display, self.identity)["checkpoint_sha256"])
        return pulse.state, row

    def finish(self, checkpoint, preview=None):
        state = self.restore(checkpoint)
        rows = []
        while int(state.vm.status) == 0:
            state, row = self.step(state, preview if not rows else None)
            rows.append(row)
        return dict(trace=rows, **self.summary(state))
