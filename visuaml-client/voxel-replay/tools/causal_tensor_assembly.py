"""A bounded structured-token causal decoder for the synthesized tool ISA."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import relu

OUT = Path(__file__).resolve().parents[1] / "evidence/causal-tensor-assembly"
CAPACITY = 16
CONTEXT = 64
MODULUS = 1 << 32


class CausalAssembly(nn.Module):
    """Decode one record from its causal prefix, with no mutable VM state."""

    def __init__(self):
        super().__init__()
        self.register_buffer("positions", 12 * torch.arange(64, dtype=torch.float64))

    def select(self, query: torch.Tensor, keys: torch.Tensor, payload: torch.Tensor):
        # Dot product gathers a matching record. Positional score breaks ties by recency.
        scores = 768 * (keys @ query) + self.positions[: keys.size(0)]
        weighted = torch.softmax(scores, dim=0) @ payload
        return 2 * (relu(weighted - 0.25) - relu(weighted - 0.75))

    def gate(self, bits: torch.Tensor, active: torch.Tensor):
        return relu(bits + active - 1)

    def add(self, left: torch.Tensor, right: torch.Tensor):
        # Five stages of parallel prefix carry on 32 little-endian bit lanes.
        carry = self.gate(left, right)
        propagate = left + right - 2 * carry
        prefix = propagate
        for distance in (1, 2, 4, 8, 16):
            cross = self.gate(prefix[distance:], carry[:-distance])
            carry = torch.cat((carry[:distance], carry[distance:] + cross))
            prefix = torch.cat((prefix[:distance], self.gate(prefix[distance:], prefix[:-distance])))
        incoming = torch.cat((left.new_zeros(1), carry[:-1]))
        return propagate + incoming - 2 * self.gate(propagate, incoming)

    def forward(self, history: torch.Tensor, external_load: torch.Tensor | None = None):
        # Lanes: program key, opcode, register selector, memory address, immediate,
        # target PC, next PC, R0, R1, live. Writes reuse address and immediate lanes.
        pc = history[-1, 75:91]
        instruction = self.select(pc, history[:, :16], history[:, 16:75])
        op, registers = instruction[:5], instruction[5:7]
        address, immediate, target = instruction[7:11], instruction[11:43], instruction[43:59]
        r0, r1 = history[-1, 91:123], history[-1, 123:155]
        previous = self.gate(r0, registers[0]) + self.gate(r1, registers[1])
        loaded = (self.select(address, history[:, 155:159], history[:, 27:59])
                  if external_load is None else external_load)
        added = self.add(previous, immediate)
        updated = self.gate(loaded, op[1]) + self.gate(added, op[3])
        changed = op[1] + op[3]
        write_r0 = self.gate(registers[0], changed)
        write_r1 = self.gate(registers[1], changed)
        next_r0 = self.gate(updated, write_r0) + self.gate(r0, 1 - write_r0)
        next_r1 = self.gate(updated, write_r1) + self.gate(r1, 1 - write_r1)
        nonzero = previous.sum() - relu(previous.sum() - 1)
        taken = self.gate(nonzero, op[4])
        sequential = torch.cat((pc.new_zeros(1), pc[:-1]))
        next_pc = self.gate(sequential, 1 - taken - op[0]) + self.gate(target, taken) + self.gate(pc, op[0])
        write_address = self.gate(address, op[2])
        write_value = self.gate(previous, op[2])
        return torch.cat((pc.new_zeros(27), write_value, pc.new_zeros(16), next_pc,
                          next_r0, next_r1, write_address, (1 - op[0]).reshape(1)))


def bits(value):
    return torch.tensor([(value >> i) & 1 for i in range(32)], dtype=torch.float64)


def start(program, memory, counter):
    if len(program) > CAPACITY or len(memory) != 4 or not program or program[-1][0] != 0:
        raise ValueError("Invalid image shape or missing final HALT")
    if not all(type(v) is int and 0 <= v < MODULUS for v in [*memory, counter]):
        raise ValueError("Invalid input word")
    data = torch.zeros((len(program) + 5, 160), dtype=torch.float64)
    for pc, (op, reg, arg) in enumerate(program):
        if not 0 <= op <= 4 or not 0 <= reg < 2:
            raise ValueError("Unsupported instruction")
        if ((op == 0 and arg != 0) or (op in (1, 2) and not 0 <= arg < 4)
                or (op == 3 and not -(1 << 31) <= arg < (1 << 31))
                or (op == 4 and not 0 <= arg < len(program))):
            raise ValueError("Invalid immediate")
        data[pc, pc] = 1
        data[pc, 16 + op] = 1
        data[pc, 21 + reg] = 1
        if op in (1, 2):
            data[pc, 23 + arg] = 1
        if op == 3:
            data[pc, 27:59] = bits(arg % MODULUS)
        if op == 4:
            data[pc, 59 + arg] = 1
    for index, value in enumerate(memory):
        data[len(program) + index, 155 + index] = 1
        data[len(program) + index, 27:59] = bits(value)
    data[-1, 75] = 1
    data[-1, 123:155] = bits(counter)
    data[-1, 159] = 1
    return data


def word(lanes):
    return sum(int(bit) << i for i, bit in enumerate(lanes.tolist()))


def decode(model, image, fuel):
    """The driver only appends the decoder's complete record and honors emitted halt."""
    if not 0 <= fuel <= CONTEXT - image.size(0) + 1:
        raise ValueError("Source fuel exceeds causal context")
    history = image
    for _ in range(fuel):
        record = model(history)
        assert torch.all((record == 0) | (record == 1))
        history = torch.cat((history, record.unsqueeze(0)))
        if record[159] == 0:
            break
    return history[image.size(0):]


def observe(records, initial):
    """Offline audit decoder. These values never feed the execution driver."""
    trace, memory = [], initial.copy()
    for record in records:
        assert record[75:91].sum() == 1 and record[155:159].sum() <= 1
        if record[155:159].any():
            memory[int(record[155:159].argmax())] = word(record[27:59])
        trace.append([int(record[75:91].argmax()), int(record[159]), word(record[91:123]),
                      word(record[123:155]), *memory])
    return trace


def freeze():
    model = torch.jit.freeze(torch.jit.script(CausalAssembly().eval()))
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "decoder.pt"))
    (OUT / "decoder.ir").write_text(str(model.inlined_graph) + "\n")
    print(hashlib.sha256((OUT / "decoder.pt").read_bytes()).hexdigest())


def replay():
    from replay_internal_tool import oracle
    from synthesize_attention_vm import OUT as RECURRENT
    from internal_tool import prepare_program

    image_file = OUT / "decoder.pt"
    frozen = RECURRENT / "executor.pt"
    model = torch.jit.load(str(image_file))
    recurrent = torch.jit.load(str(frozen))
    cases = json.loads((OUT / "holdout.json").read_text())
    rows = []
    for case in cases:
        program, seed, counter = (case[k] for k in ("program", "seed", "counter"))
        memory = [seed, 0, 0, 0]
        image = start(program, memory, counter)
        fuel = case["fuel"]
        expected = oracle(program, [seed, counter])[:fuel]
        retired = decode(model, image, fuel)
        trace = observe(retired, memory)
        assert trace == expected, (case["name"], trace, expected)
        _, frozen_trace = recurrent(prepare_program(program), torch.tensor([seed, counter]))
        assert frozen_trace[:len(trace)].tolist() == trace
        rows.append(dict(name=case["name"], initial_records=len(image), generated_records=len(retired),
                         trace=trace, emitted=[{"next_pc": step[0], "live": step[1],
                            "write": [int(record[155:159].argmax()), word(record[27:59])]
                            if record[155:159].any() else None}
                          for step, record in zip(trace, retired)]))
    return dict(decoder_sha256=hashlib.sha256(image_file.read_bytes()).hexdigest(),
                recurrent_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(), cases=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        if args.freeze:
            freeze()
        else:
            observed = replay()
            path = OUT / "replay.json"
            if args.write:
                path.write_text(json.dumps(observed, indent=2) + "\n")
            else:
                assert observed == json.loads(path.read_text())
            print(f"Verified {len(observed['cases'])} exact contextual and recurrent traces")
