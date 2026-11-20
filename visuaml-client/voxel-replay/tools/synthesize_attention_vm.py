"""Lower an affine/ReLU transition graph to a frozen recurrent hard-attention block."""
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
import json

import torch
from torch import nn

OUT = Path(__file__).resolve().parents[1] / "evidence/synthesized-attention"
CAPACITY = 48
U32 = 1 << 32


@dataclass
class Affine:
    terms: dict = field(default_factory=dict)
    bias: int = 0

    def __add__(self, other):
        if isinstance(other, int):
            return Affine(self.terms.copy(), self.bias + other)
        terms = self.terms.copy()
        for index, coefficient in other.terms.items():
            terms[index] = terms.get(index, 0) + coefficient
        return Affine({k: v for k, v in terms.items() if v}, self.bias + other.bias)

    __radd__ = __add__

    def __mul__(self, scale):
        return Affine({k: v*scale for k, v in self.terms.items() if v*scale}, self.bias*scale)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + other * -1


class ResidualFFN(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x):
        return torch.cat((x, torch.relu(self.weight @ x + self.bias)))


def compile_transition():
    # Residual inputs: pc, running, two registers, four memory words,
    # fetched opcode/register/immediate and attended operand/memory value.
    width = 13
    slots = [Affine({i: 1}) for i in range(width)]
    relus = []
    levels = dict.fromkeys(range(width), 0)

    def relu(x):
        index = width + len(relus)
        levels[index] = 1 + max((levels[i] for i in x.terms), default=0)
        relus.append(x)
        return Affine({index: 1})

    def step(x):
        # Indicator x >= 0 on integer inputs.
        return relu(relu(x+1) - relu(x))

    def eq(x, k):
        return step(x-k) - step(x-k-1)

    def both(*conditions):
        return relu(sum(conditions) - (len(conditions)-1))

    def select(value, bit, bound):
        # Exact for 0 <= value <= bound and bit in {0,1}.
        return relu(value + bound*(bit-1))

    pc, live, r0, r1, m0, m1, m2, m3, op, reg, imm, old, loaded = slots
    load, store, add, jump = (eq(op, k) for k in (1, 2, 3, 4))
    running = both(live, Affine(bias=1)-eq(op, 0))
    arithmetic = old + imm
    wrapped = arithmetic + U32*(Affine(bias=1)-step(arithmetic)) - U32*step(arithmetic-U32)
    updated_regs = []
    for i, value in enumerate((r0, r1)):
        use_load = both(live, load, eq(reg, i))
        use_add = both(live, add, eq(reg, i))
        updated_regs.append(
            select(value, Affine(bias=1)-use_load-use_add, U32)
            + select(loaded, use_load, U32) + select(wrapped, use_add, U32))
    updated_memory = []
    for i, value in enumerate((m0, m1, m2, m3)):
        write = both(live, store, eq(imm, i))
        updated_memory.append(select(value, Affine(bias=1)-write, U32) + select(old, write, U32))
    taken = both(running, jump, step(old-1))
    # Inactive jump inputs can carry a signed immediate. Shift to a
    # nonnegative domain for the exact selector and cancel the shift.
    target = select(imm+(1 << 31), taken, 1 << 32) - (1 << 31)*taken
    next_pc = (select(pc, Affine(bias=1)-running, CAPACITY)
               + select(pc+1, running-taken, CAPACITY) + target)
    output = [next_pc, running, *updated_regs, *updated_memory]

    def matrix(expressions, mapping):
        weights = torch.zeros((len(expressions), len(mapping)), dtype=torch.float64)
        for row, expression in enumerate(expressions):
            for node, coefficient in expression.terms.items():
                weights[row, mapping[node]] = coefficient
        return weights, torch.tensor([x.bias for x in expressions], dtype=torch.float64)

    mapping = {i: i for i in range(width)}
    layers = []
    for level in range(1, max(levels.values())+1):
        nodes = [i for i in range(width, width+len(relus)) if levels[i] == level]
        layers.append(ResidualFFN(*matrix([relus[i-width] for i in nodes], mapping)))
        for i in nodes:
            mapping[i] = len(mapping)
    return nn.ModuleList(layers), matrix(output, mapping), dict(
        inputs=width, relu_nodes=len(relus), layer_widths=[len(x.bias) for x in layers])


class HardHead(nn.Module):
    def __init__(self, capacity, value_width):
        super().__init__()
        pos = torch.arange(capacity, dtype=torch.float64)
        self.register_buffer("positions", pos)
        self.register_buffer("key_basis", torch.stack((torch.ones_like(pos), pos, pos.square()), dim=1))
        self.register_buffer("query_projection", torch.eye(2, dtype=torch.float64))
        self.register_buffer("key_projection", torch.tensor([[0, 2, 0], [0, 0, -1]], dtype=torch.float64))
        self.register_buffer("value_projection", torch.eye(value_width, dtype=torch.float64))

    def forward(self, address: torch.Tensor, values: torch.Tensor):
        count = values.size(0)
        q = self.query_projection @ torch.stack((address, torch.ones_like(address)))
        keys = self.key_basis[:count] @ self.key_projection.T
        winner = (keys @ q).argmax()
        selection = (self.positions[:count] == winner).to(torch.float64)
        return selection @ (values @ self.value_projection.T)


class SynthesizedVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fetch = HardHead(CAPACITY, 3)
        self.reg_read = HardHead(2, 1)
        self.mem_read = HardHead(4, 1)
        self.layers, (weight, bias), self.layout = compile_transition()
        self.register_buffer("output_weight", weight)
        self.register_buffer("output_bias", bias)

    def forward(self, program: torch.Tensor, operands: torch.Tensor):
        state = torch.cat((torch.tensor([0, 1, 0], dtype=torch.float64),
                           operands[1:].to(torch.float64), operands[:1].to(torch.float64),
                           torch.zeros(3, dtype=torch.float64)))
        program = program.to(torch.float64)
        trace = torch.jit.annotate(list[torch.Tensor], [])
        for _ in range(48):
            instruction = self.fetch(state[0], program)
            old = self.reg_read(instruction[1], state[2:4].reshape(2, 1))
            loaded = self.mem_read(instruction[2], state[4:].reshape(4, 1))
            residual = torch.cat((state, instruction, old, loaded))
            for layer in self.layers:
                residual = layer(residual)
            state = self.output_weight @ residual + self.output_bias
            trace.append(state.to(torch.int64))
        return state[4].to(torch.int64), torch.stack(trace)


def freeze():
    torch.set_num_threads(1)
    start = perf_counter_ns()
    model = SynthesizedVM().eval()
    synthesis_ns = perf_counter_ns()-start
    frozen = torch.jit.freeze(torch.jit.script(model))
    frozen.save(str(OUT / "executor.pt"))
    # Data-only export of all synthesized projection and FFN coefficients.
    weights = {name: tensor.tolist() for name, tensor in model.named_buffers()}
    (OUT / "weights.json").write_text(json.dumps(dict(layout=model.layout, weights=weights),
                                                separators=(",", ":"))+"\n")
    (OUT / "build.json").write_text(json.dumps(dict(synthesis_ns=synthesis_ns,
         torch=str(torch.__version__), operators=sorted(torch.jit.export_opnames(frozen))), indent=2)+"\n")


if __name__ == "__main__":
    freeze()
