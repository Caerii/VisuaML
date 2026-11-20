"""Forkable passage execution at committed WASM source boundaries."""
from dataclasses import dataclass, field
import struct

from frozen_call_checkpoint import canonical, digest, encode, restore as restore_vm
from replay_voxel_call_checkpoint import CELL, FUEL, Witness, page, words
from visuaml.compiler_lab.frozen_calls_v58 import CallState

SEMANTICS = "frozen-direct-call/passage-events-v1"


@dataclass
class Run:
    witness: Witness
    vm: CallState
    inputs: tuple
    next_event: int
    retired: int
    outputs: list = field(default_factory=list)

    def step(self, trace=False):
        if int(self.vm.vm.status) != 0 or int(self.vm.vm.fuel) == 0:
            raise ValueError("A source step requires a running invocation with fuel")
        if trace:
            self.vm, record = self.witness.step(self.vm)
        else:
            self.vm = self.witness.machine(self.witness.image.rows, self.vm).state
            record = None
        self.retired += 1
        return record

    def finish_tick(self, trace=False):
        if self.next_event != len(self.outputs) + 1:
            raise ValueError("No pending tick")
        records = []
        while int(self.vm.vm.status) == 0:
            item = self.step(trace)
            if trace:
                records.append(item)
        assert int(self.vm.vm.status) == 1 and int(self.vm.depth) == 0
        memory = page(self.vm)
        terrain, _, position, _, observed, commits = words(memory)
        output = dict(tick=self.next_event - 1, world=[terrain, position, observed, commits],
                      returned=int(self.vm.vm.stack[0]), page_sha256=digest(memory),
                      retired=self.retired)
        self.outputs.append(output)
        return output, records

    def begin_tick(self):
        if self.next_event != len(self.outputs) or self.next_event >= len(self.inputs):
            raise ValueError("Input cannot be presented at this boundary")
        memory = bytearray(page(self.vm))
        struct.pack_into("<i", memory, CELL + 4, 0)
        struct.pack_into("<i", memory, CELL + 12, self.inputs[self.next_event])
        self.vm = self.witness.start(memory)
        self.next_event += 1

    def tick(self, trace=False):
        self.begin_tick()
        return self.finish_tick(trace)


def seal(payload):
    return dict(payload, sha256=digest(canonical(payload)))


def export_state(run, after_source_step):
    if type(after_source_step) is not int or after_source_step < run.retired:
        raise ValueError("Checkpoint boundary precedes the current source position")
    while run.retired < after_source_step:
        run.step()
    if int(run.vm.vm.status) != 0 or run.next_event != len(run.outputs) + 1:
        raise ValueError("Checkpoint requires a suspended invocation")
    inputs = list(run.inputs)
    return seal(dict(format_version=1, semantics=SEMANTICS,
                     vm=encode(run.vm, run.witness.identity), retired_source_steps=run.retired,
                     inputs=inputs, input_log_sha256=digest(canonical(inputs)),
                     next_event=run.next_event, pending_outputs=json_copy(run.outputs)))


def restore(witness, checkpoint):
    payload = {key: value for key, value in checkpoint.items() if key != "sha256"}
    if payload.get("format_version") != 1 or payload.get("semantics") != SEMANTICS:
        raise ValueError("Checkpoint execution semantics mismatch")
    if seal(payload)["sha256"] != checkpoint.get("sha256"):
        raise ValueError("Checkpoint integrity mismatch")
    inputs = payload["inputs"]
    if (not isinstance(inputs, list) or not inputs
            or any(type(event) is not int or event not in (0, 1) for event in inputs)
            or digest(canonical(inputs)) != payload["input_log_sha256"]):
        raise ValueError("Checkpoint input log mismatch")
    cursor, retired, outputs = (payload[key] for key in ("next_event", "retired_source_steps", "pending_outputs"))
    if (type(cursor) is not int or not 1 <= cursor <= len(inputs) or type(retired) is not int or retired < 1
            or not isinstance(outputs, list) or len(outputs) != cursor - 1):
        raise ValueError("Checkpoint event boundary mismatch")
    vm = restore_vm(payload["vm"], witness.identity, witness.image)
    active_steps = FUEL - int(vm.vm.fuel)
    if any(not isinstance(record, dict) or record.get("tick") != index
           or type(record.get("retired")) is not int or not 0 < record["retired"] <= retired
           or not isinstance(record.get("world"), list) or len(record["world"]) != 4
           for index, record in enumerate(outputs)):
        raise ValueError("Checkpoint pending output mismatch")
    previous_steps = outputs[-1]["retired"] if outputs else 0
    if not 0 <= active_steps <= FUEL or previous_steps + active_steps != retired:
        raise ValueError("Checkpoint source count and fuel mismatch")
    return Run(witness, vm, tuple(inputs), cursor, retired, json_copy(outputs))


def json_copy(outputs):
    # Copies scalar output records and their one mutable world vector.
    return [{**record, "world": record["world"].copy()} for record in outputs]


def fork(witness, checkpoint, future_input_override):
    run = restore(witness, checkpoint)
    inputs = list(run.inputs)
    for index, event in future_input_override.items():
        if (type(index) is not int or not run.next_event <= index < len(inputs)
                or type(event) is not int or event not in (0, 1)):
            raise ValueError("Override must select a future event and a legal action")
        inputs[index] = event
    run.inputs = tuple(inputs)
    return run
