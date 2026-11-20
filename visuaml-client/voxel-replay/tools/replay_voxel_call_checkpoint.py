"""Resume a Rust helper after its committed terrain store."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from frozen_call_checkpoint import canonical, digest, encode, identities, restore
from replay_frozen_calls import OUT as CALLS
from replay_frozen_recurrent import PACKED, wasmtime_run
from visuaml.compiler_lab.frozen_calls_v58 import initialize, load_call_image

OUT = CALLS.parent / "voxel-call-checkpoint"
CELL = 65512
FUEL = 2000


def page(state):
    return state.vm.page.numpy().tobytes()


def words(raw):
    return list(struct.unpack_from("<6i", raw, CELL))


class Witness:
    def __init__(self):
        torch.set_num_threads(1)
        self.raw = (OUT / "passage.wasm").read_bytes()
        self.image = load_call_image(self.raw)
        graph = CALLS / "machine.pt"
        self.machine = torch.jit.load(str(graph))
        self.identity = identities(graph.read_bytes(), self.image)

    def start(self, raw):
        return initialize(self.image, CELL, raw, FUEL)

    def step(self, state):
        pc = int(state.vm.pc)
        function = int(pc >= self.image.starts[1])
        opcode = self.image.functions[function][pc-self.image.starts[function]]
        pulse = self.machine(self.image.rows, state)
        state = pulse.state
        boundary = encode(state, self.identity)
        return state, dict(pc=pc, function=function, opcode=opcode, access=pulse.access.tolist(),
                           next_pc=int(state.vm.pc), fuel=int(state.vm.fuel), depth=int(state.depth),
                           locals=state.vm.locals.tolist(), stack=state.vm.stack[:int(state.vm.height)].tolist(),
                           state_sha256=boundary["checkpoint_sha256"])

    def finish(self, state):
        trace = []
        while int(state.vm.status) == 0:
            state, record = self.step(state)
            trace.append(record)
        assert int(state.vm.status) == 1 and int(state.depth) == 0
        return state, trace

    def export(self):
        initial = bytearray(65536)
        struct.pack_into("<6i", initial, CELL, 7, 1, 0, 0, -1, 0)
        state = self.start(initial)
        prefix = []
        while int(state.vm.status) == 0:
            state, record = self.step(state)
            prefix.append(record)
            if record["access"] == [2, CELL, 7, 0]:
                assert int(state.depth) == 1 and state.vm.locals[1].item() == 7
                assert words(page(state)) == [0, 1, 0, 0, -1, 0]
                return bytes(initial), encode(state, self.identity), prefix, state
        raise AssertionError("No in-helper clear")

    def next_tick(self, completed, move):
        initial = bytearray(page(completed))
        struct.pack_into("<i", initial, CELL + 4, 0)
        struct.pack_into("<i", initial, CELL + 12, move)
        final, trace = self.finish(self.start(initial))
        assert wasmtime_run(self.raw, CELL, initial) == (move, page(final))
        assert words(page(final)) == [0, 0, move, move, 0, 2]
        assert sum(row["opcode"] == "call" for row in trace) == 1
        return dict(move_intent=move, record=words(page(final)), trace=trace, page_sha256=digest(page(final)))

    def resume(self, checkpoint, move):
        state = restore(checkpoint, self.identity, self.image)
        assert encode(state, self.identity) == checkpoint
        final, suffix = self.finish(state)
        assert not any(row["opcode"] == "call" or row["access"][:2] == [2, CELL] for row in suffix)
        returns = [row for row in suffix if row["function"] == 0 and row["depth"] == 0]
        assert len(returns) == 1 and returns[0]["stack"] == [CELL, 7]
        assert words(page(final)) == [0, 1, 0, 0, 7, 1]
        assert sum(row["access"][:2] == [2, CELL+20] for row in suffix) == 1
        return dict(suffix=suffix, completed=words(page(final)), page_sha256=digest(page(final)),
                    next_tick=self.next_tick(final, move))

    def rejection_checks(self, checkpoint):
        for key in self.identity:
            wrong = dict(checkpoint, identities={**self.identity, key: "wrong"})
            try:
                restore(wrong, self.identity, self.image)
            except ValueError as error:
                assert "identity mismatch" in str(error)
            else:
                raise AssertionError("Mismatched identity accepted")
        damaged = dict(checkpoint, depth=0)
        try:
            restore(damaged, self.identity, self.image)
        except ValueError as error:
            assert "integrity mismatch" in str(error)
        else:
            raise AssertionError("Damaged checkpoint accepted")

        malformed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
        malformed["registers"] = {**malformed["registers"], "stack": [7]}
        malformed["checkpoint_sha256"] = digest(canonical(malformed))
        try:
            restore(malformed, self.identity, self.image)
        except ValueError as error:
            assert "register shape" in str(error)
        else:
            raise AssertionError("Invalid stack shape accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--move", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    witness = Witness()
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        if args.resume:
            print(json.dumps(witness.resume(json.loads(args.resume.read_text()), args.move)))
            return
        initial, checkpoint, prefix, state = witness.export()
        if args.export:
            args.export.write_bytes(canonical(checkpoint) + b"\n")
            return
        witness.rejection_checks(checkpoint)
        uninterrupted, suffix = witness.finish(state)
        assert wasmtime_run(witness.raw, CELL, initial) == (0, page(uninterrupted))

        with TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            command = [sys.executable, str(Path(__file__).resolve())]
            subprocess.check_call([*command, "--export", str(path)])
            assert json.loads(path.read_text()) == checkpoint
            copies = {}
            for move in (0, 1):
                copies[str(move)] = json.loads(subprocess.check_output(
                    [*command, "--resume", str(path), "--move", str(move)]))
                assert copies[str(move)]["suffix"] == suffix
                assert copies[str(move)]["page_sha256"] == digest(page(uninterrupted))
                assert copies[str(move)]["next_tick"] == witness.next_tick(uninterrupted, move)

        # The suspended caller has values beneath the call argument. Also round
        # trip a later boundary where the active helper has an operand.
        later = restore(checkpoint, witness.identity, witness.image)
        later, _ = witness.step(later)
        later, _ = witness.step(later)
        assert int(later.vm.height) == 1 and int(later.vm.stack[0]) == 7
        live = encode(later, witness.identity)
        _, remaining = witness.finish(restore(live, witness.identity, witness.image))
        assert remaining == suffix[2:]

    replay = dict(identities=witness.identity, packed_fallback=False, cell_address=CELL,
                  checkpoint_after=len(prefix), prefix=prefix, suffix=suffix,
                  observer_return=7, copies={key: {k: v for k, v in copy.items() if k != "suffix"}
                                              for key, copy in copies.items()})
    for name, data in (("checkpoint.json", checkpoint), ("replay.json", replay)):
        path = OUT / name
        if args.write:
            path.write_bytes(canonical(data)+b"\n" if name == "checkpoint.json"
                             else (json.dumps(data, indent=2)+"\n").encode())
        else:
            assert json.loads(path.read_text()) == data
    print(f"Checkpoint after step {len(prefix)}; {len(suffix)} identical resumed source boundaries per copy.")


if __name__ == "__main__":
    main()
