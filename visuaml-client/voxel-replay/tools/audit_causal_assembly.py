"""Differentially exercise the frozen causal decoder over admitted bytecode images."""
import argparse
import json
from random import Random
from time import perf_counter_ns

import torch

from causal_tensor_assembly import CONTEXT, MODULUS, OUT, decode, observe, start
from frozen_call_checkpoint import digest
from internal_tool import prepare_program
from synthesize_attention_vm import OUT as RECURRENT


def cases():
    rng = Random(41151)
    immediates = [-(1 << 31), -1, 0, 1, 3, (1 << 31) - 1]
    words = [0, 1, (1 << 31) - 1, 1 << 31, MODULUS - 2, MODULUS - 1]
    for index in range(192):
        length = rng.randrange(2, 17)
        code = []
        for _ in range(length - 1):
            op, register = rng.randrange(1, 5), rng.randrange(2)
            if op in (1, 2):
                operand = rng.randrange(4)
            elif op == 3:
                operand = rng.choice(immediates)
            else:
                operand = rng.randrange(length)
            code.append([op, register, operand])
        code.append([0, 0, 0])
        yield code, rng.choice(words), index % 10, rng.randrange(49)


def reference(code, initial, count, fuel):
    pc, registers, live = 0, [0, count], 1
    memory = initial.copy()
    trace = []
    for _ in range(fuel):
        op, register, operand = code[pc]
        if op == 0:
            live = 0
        elif op == 4 and registers[register]:
            pc = operand
        else:
            if op == 1:
                registers[register] = memory[operand]
            elif op == 2:
                memory[operand] = registers[register]
            elif op == 3:
                registers[register] = (registers[register] + operand) % MODULUS
            pc += 1
        trace.append([pc, live, *registers, *memory])
        if not live:
            break
    return trace


def run():
    decoder_path = OUT / "decoder.pt"
    decoder = torch.jit.load(str(decoder_path))
    recurrent = torch.jit.load(str(RECURRENT / "executor.pt"))
    retired = 0
    for index, (code, seed, count, fuel) in enumerate(cases()):
        memory = [seed, 0, 0, 0]
        image = start(code, memory, count)
        budget = min(fuel, CONTEXT - image.size(0))
        actual = observe(decode(decoder, image, budget), memory)
        exact = reference(code, memory, count, budget)
        assert actual == exact, (index, code, seed, count, budget, actual, exact)
        _, recurrent_trace = recurrent(prepare_program(code), torch.tensor([seed, count]))
        assert actual == recurrent_trace[:len(actual)].tolist()
        retired += len(actual)
    return dict(decoder_sha256=digest(decoder_path.read_bytes()), generator_seed=41151,
                program_count=index + 1, checked_boundaries=retired)


def benchmark():
    path = OUT / "decoder.pt"
    start_time = perf_counter_ns()
    model = torch.jit.load(str(path))
    load_ns = perf_counter_ns() - start_time
    samples = {}
    for case in json.loads((OUT / "holdout.json").read_text()):
        image = start(case["program"], [case["seed"], 0, 0, 0], case["counter"])
        records = decode(model, image, case["fuel"])
        elapsed = []
        for _ in range(12):
            start_time = perf_counter_ns()
            decode(model, image, case["fuel"])
            elapsed.append(perf_counter_ns() - start_time)
        record_bytes = image.size(1) * image.element_size()
        samples[case["name"]] = dict(raw_ns=elapsed, initial_records=image.size(0),
                                     emitted_records=len(records),
                                     final_history_bytes=(image.size(0)+len(records))*record_bytes)
    return dict(decoder_sha256=digest(path.read_bytes()), load_ns=load_ns,
                intraop_threads=torch.get_num_threads(), cases=samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        print(json.dumps(benchmark() if args.benchmark else run(), indent=2))
