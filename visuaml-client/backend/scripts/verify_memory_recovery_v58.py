"""Recover and verify a bounded in-place fold from original WASM memory effects."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path

import wasmtime

from scripts.benchmark_neural_vm_v58 import git_identity
from scripts.verify_packed_control_v58 import GATES, first_divergence, prepare
from visuaml.compiler_lab.neural_microstep_executor_v58 import MemoryWeights
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    MEMORY_BYTES, ControlOperation, ControlProgram, MemoryArgument, decode_control, encode_control,
)
from visuaml.compiler_lab.wasm_research_corpus_v57 import op

DOMAIN_VALUES = tuple(range(-2, 3))
MAX_LENGTH = 4
CANDIDATES = {
    "overwrite": ("value", lambda acc, value: value),
    "sum": ("signed_i32(acc + value)", lambda acc, value: (acc + value + (1 << 31)) % (1 << 32) - (1 << 31)),
    "maximum": ("max(acc, value)", max),
}


def render(candidate):
    expression, _ = CANDIDATES[candidate]
    return (
        "acc = 0\nfor i in range(length):\n"
        "    value = read_i32_le(memory, 4*i)\n"
        f"    acc = {expression}\n"
        "    write_i32_le(memory, 4*i, acc)\nreturn acc"
    )


def prefix_program():
    # Keep a copy of the destination address below the loaded word.
    address = (op("local.get", 1), op("i32.const", 4), op("i32.mul"))
    return ControlProgram(2, (
        op("block"), op("loop"), op("local.get", 1), op("local.get", 0),
        op("i32.lt_s"), op("i32.eqz"), op("br_if", 1),
        *address, *address, ControlOperation("i32.load", MemoryArgument()),
        op("local.get", 2), op("i32.add"),
        ControlOperation("i32.store", MemoryArgument()),
        *address, ControlOperation("i32.load", MemoryArgument()), op("local.set", 2),
        op("local.get", 1), op("i32.const", 1), op("i32.add"), op("local.set", 1),
        op("br", 0), op("end"), op("end"), op("local.get", 2), op("end"),
    ))


def initial_page(values):
    page = bytearray(b"\xa5" * MEMORY_BYTES)
    for index, value in enumerate(values):
        struct.pack_into("<i", page, 4 * index, value)
    return page


def word_values(page, count):
    return tuple(struct.unpack_from("<i", page, 4 * index)[0] for index in range(count))


def symbolic(candidate, values):
    page = initial_page(values)
    acc = 0
    _, step = CANDIDATES[candidate]
    for index, value in enumerate(values):
        acc = step(acc, value)
        struct.pack_into("<i", page, 4 * index, acc)
    return acc, page


def replay_trace(initial, trace):
    """Apply the logged accesses and fail at the first inconsistent read or write."""
    page = bytearray(initial)
    accesses = 0
    for row in trace:
        access = row["memory_access"]
        if access is None:
            continue
        accesses += 1
        address = access["address"]
        before = bytes.fromhex(access["before_hex"])
        after = bytes.fromhex(access["after_hex"])
        if (not 0 <= address <= len(page) - 4 or len(before) != 4 or len(after) != 4
                or page[address:address + 4] != before):
            raise AssertionError(("inconsistent memory access", row["step"], access))
        if access["kind"] == "i32.store":
            page[address:address + 4] = after
        elif access["kind"] != "i32.load" or before != after:
            raise AssertionError(("invalid load trace", row["step"], access))
    return page, accesses


def collect(root):
    source = git_identity(root)
    program = prefix_program()
    raw = encode_control(program)
    assert decode_control(raw) == program
    native = prepare(program)
    store = wasmtime.Store()
    instance = wasmtime.Instance(store, wasmtime.Module(store.engine, raw), []).exports(store)

    def original(values, *, traced=False):
        initial = initial_page(values)
        instance["memory"].write(store, initial, 0)
        result = instance["run"](store, len(values))
        expected = instance["memory"].read(store)
        observed = bytearray(initial)
        executed = native.run_memory(len(values), observed, capture_trace=traced)
        assert (result, expected) == (executed.result_i32, observed), values
        if traced:
            replayed, _ = replay_trace(initial, executed.trace)
            assert replayed == observed
        return executed, expected

    # A singleton has multiple explanations. Search the declared grammar
    # and refute the first fitting hypothesis with a domain witness.
    seeds = [(3,)]
    expected = {}

    def observe(values):
        result, page = original(values)
        value = result.result_i32, word_values(page, len(values))
        expected[values] = value
        return value

    def predict(candidate, values):
        result, page = symbolic(candidate, values)
        return result, word_values(page, len(values))

    observe(seeds[0])
    domain = tuple(values for count in range(MAX_LENGTH + 1)
                   for values in itertools.product(DOMAIN_VALUES, repeat=count))
    for values in domain:
        observe(values)
    rounds = []
    while True:
        candidate = next(name for name in CANDIDATES if all(predict(name, x) == expected[x] for x in seeds))
        counterexample = next((values for values in domain if predict(candidate, values) != expected[values]), None)
        if counterexample is None:
            rounds.append({"candidate": candidate, "counterexample": None})
            break
        rounds.append({"candidate": candidate, "counterexample": counterexample,
                       "candidate_output": predict(candidate, counterexample),
                       "original_output": expected[counterexample]})
        seeds.append(counterexample)

    assert candidate == "sum" and len(rounds) > 1
    # Check entire pages against the recovered program as well as against
    # the native execution. Untouched sentinel bytes are part of the claim.
    for values in domain:
        resolved, page = original(values)
        assert (resolved.result_i32, page) == symbolic(candidate, values)

    showcase = (3, -1, 4)
    nominal, end_page = original(showcase, traced=True)
    ablated_page = initial_page(showcase)
    ablated = prepare(program, memory_fixed=MemoryWeights(store_gain=0)).run_memory(3, ablated_page)
    difference = first_divergence(nominal.trace, ablated.trace, (*GATES, "memory_access"))
    assert difference["changes"]["memory_access"][0]["kind"] == "i32.store"
    assert word_values(end_page, 3) == (3, 2, 6)
    assert word_values(ablated_page, 3) == showcase and ablated.memory_weights_sha256 != nominal.memory_weights_sha256
    return {
        "source": source, "wasm_hex": raw.hex(), "wasm_sha256": hashlib.sha256(raw).hexdigest(),
        "domain": {"length_min": 0, "length_max": MAX_LENGTH, "values": DOMAIN_VALUES, "cases": len(domain)},
        "grammar": tuple(CANDIDATES), "rounds": rounds, "recovered_symbolic_program": render(candidate),
        "showcase": {"input": showcase, "result": nominal.result_i32, "output": word_values(end_page, 3),
                     "trace": nominal.trace, "memory_sha256": hashlib.sha256(end_page).hexdigest()},
        "store_gain_intervention": {"first_divergence": difference, "output": word_values(ablated_page, 3)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Use a fresh output path outside the checkout")
    evidence = collect(root)
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
