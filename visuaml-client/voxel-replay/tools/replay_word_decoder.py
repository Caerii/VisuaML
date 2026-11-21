"""Audit word-level generated instructions against independent source states."""
import argparse
import json
from random import Random
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from causal_tensor_assembly import OUT as STRUCTURED, decode, observe, start
from frozen_call_checkpoint import digest
from replay_decoder_execution_loop import run
from vocabulary_tensor_assembly import OUT as VOCABULARY, FRAME as OLD_FRAME, encode_image, generate
from word_decoder import OUT, FRAME, COMMIT, HALT, EXEC, CODE, NONE, PAD, OPERATIONS, WordCircuit

MASK = (1 << 32)-1


def pack(program, memory):
    start(program, memory, 0)
    rows = []
    for pc, (op, reg, operand) in enumerate(program):
        if reg != 0:
            raise ValueError("The word decoder only supports R0")
        immediate = operand & MASK if op == 3 else 0
        rows.extend([PAD, pc, 256+op, PAD, PAD, PAD, 0 if op == 3 else operand,
                     *immediate.to_bytes(4, "little"), CODE])
    for address, value in enumerate(memory):
        rows.extend([PAD, 0, 0, 0, 0, 0, address, *value.to_bytes(4, "little"), COMMIT])
    return torch.tensor(rows + [PAD, 0, 0, 0, 0, 0, NONE, 0, 0, 0, 0, COMMIT])


def audit(tokens, initial):
    memory = initial.copy()
    states, writes = [], []
    for index in range(0, len(tokens)-FRAME+1, FRAME):
        row = tokens[index:index+FRAME]
        assert row[0] == EXEC and row[-1] in (COMMIT, HALT)
        address = row[6]
        if address != NONE:
            assert address in range(4) and row[-1] == COMMIT
            value = int.from_bytes(bytes(row[7:11]), "little")
            memory[address] = value
            writes.append(dict(step=len(states)+1, token=index+FRAME, address=address, value=value))
        states.append([row[1], int(row[-1] != HALT), int.from_bytes(bytes(row[2:6]), "little"), 0, *memory])
    return states, writes


def alu_check():
    rng = Random(9243)
    edges = [0, 1, 2, 3, 0xfffffffe, 0xffffffff, 0x7fffffff, 0x80000000]
    cases = [(x, y) for x in edges for y in edges]
    cases.extend((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(2048))
    pair = torch.tensor(cases)
    weights = 2**torch.arange(32)
    lanes = ((pair[:, :, None] // weights) % 2).double()
    add, test = WordCircuit(OPERATIONS)(lanes[:, 0], lanes[:, 1])
    assert torch.equal((add.long()*weights).sum(1), (pair[:, 0]+pair[:, 1]) & MASK)
    assert torch.equal(test[:, 0], (pair[:, 0] == 0).double())
    return len(cases)


def replay(measure):
    path = OUT / "decoder.pt"
    model = torch.jit.load(str(path))
    structured = torch.jit.load(str(STRUCTURED / "decoder.pt"))
    vocabulary = torch.jit.load(str(VOCABULARY / "decoder.pt"))
    spec = json.loads((OUT / "holdout.json").read_text())
    result, timing = {}, {}
    for name, fixture in spec["cases"].items():
        program, memory = fixture["program"], fixture["memory"]
        prefix = pack(program, memory)
        budget = (512-len(prefix))//FRAME*FRAME
        cached = lambda: run(model, prefix, True, budget=budget, halt_token=HALT)
        full = lambda: run(model, prefix, False, budget=budget, halt_token=HALT)
        tokens, _, _, cache = cached()
        assert tokens == full()[0] and tokens[-1] == HALT
        states, writes = audit(tokens, memory)
        steps = len(states)
        assert states == reference(program, memory, 0, steps)
        image = start(program, memory, 0)
        packed = encode_image(image)
        direct = lambda: decode(structured, image, steps)
        previous = lambda: generate(vocabulary, packed, steps*OLD_FRAME)
        assert states == observe(direct(), memory) == observe(vocabulary.committed(previous()), memory)
        result[name] = dict(prefix=prefix.tolist(), tokens=tokens, states=states, writes=writes,
                            kv_bytes=sum(t.numel()*t.element_size() for layer in cache for t in layer),
                            cache_positions=cache[0][0].size(1))
        if measure:
            calls = [("cached", cached), ("full", full), ("structured", direct), ("prior_vocabulary", previous)]
            samples = {}
            for _, call in calls:
                call()
            for n in range(8):
                for label, call in calls[::1 if n % 2 == 0 else -1]:
                    began = perf_counter_ns()
                    outcome = call()
                    sample = dict(total_ns=perf_counter_ns()-began)
                    if label in ("cached", "full"):
                        sample.update(prefill_ns=outcome[1], decode_ns=outcome[2])
                    samples.setdefault(label, []).append(sample)
            timing[name] = samples
    assert result["wrap-one"]["states"][4][0] == 7 and result["wrap-zero"]["states"][4][0] == 5
    assert result["wrap-one"]["writes"][0]["value"] == 1 and result["wrap-zero"]["writes"][0]["value"] == 0

    config = spec["intervention"]
    base = result[config["case"]]
    initial = spec["cases"][config["case"]]["memory"]
    commit = config["store_step"] * FRAME
    assert base["writes"][0]["value"] == config["original"]
    pref = torch.tensor(base["prefix"])
    begin = commit-FRAME
    for cut in range(begin+1, commit):
        partial, _, _, cache = run(model, pref, True, budget=cut, halt_token=HALT)
        assert partial == base["tokens"][:cut] and audit(partial, initial)[0][-1][4] == initial[0]
        assert not cache[2][0][0, len(pref)+begin:, :4].any()
        assert not cache[2][1][0, len(pref)+begin:].any()
    change_at = begin+7
    changed, _, _, _ = run(model, pref, True, budget=300, substitute=(change_at, config["replacement"]),
                           halt_token=HALT)
    assert changed == run(model, pref, False, budget=300, substitute=(change_at, config["replacement"]),
                          halt_token=HALT)[0]
    states, writes = audit(changed, initial)
    assert states[:2] == base["states"][:2] and states[2][4] == config["replacement"]
    assert states[config["next_add_step"]-1][2] == config["next_add_result"]
    assert states[config["branch_step"]-1][0] == config["exit_pc"] and changed[-1] == HALT
    tail = reference(spec["cases"][config["case"]]["program"], [1,0,0,0], 0, 5)
    assert states[4:] == tail
    stress_program = spec["cases"]["countdown"]["program"]
    stress_memory = [8, 0, 0, 0]
    stress_prefix = pack(stress_program, stress_memory)
    limit = 512-len(stress_prefix)
    stress, _, _, stress_cache = run(model, stress_prefix, True, budget=limit, halt_token=HALT)
    full_logits, _ = model(torch.cat((stress_prefix, torch.tensor(stress))), [])
    assert full_logits[len(stress_prefix)-1:-1].argmax(dim=-1).tolist() == stress
    stress_states, _ = audit(stress, stress_memory)
    assert stress_states == reference(stress_program, stress_memory, 0, limit//FRAME)
    assert stress_cache[0][0].size(1) == 512
    if measure:
        (OUT / "timings.json").write_text(json.dumps(timing, indent=2)+"\n")
    return dict(profile="visuaml-word-decoder-v1", decoder_sha256=digest(path.read_bytes()),
                holdout=spec, cases=result, context_probe=dict(initial_memory=stress_memory,
                tokens=stress, source_steps=len(stress_states), context_tokens=512), alu_pairs=alu_check(), partial_store_lengths=FRAME-1,
                intervention=dict(tokens=changed, states=states, writes=writes, changed_token=change_at+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        observed = replay(args.write)
    archive = OUT / "replay.json"
    if args.write:
        archive.write_text(json.dumps(observed, indent=2)+"\n")
    else:
        assert observed == json.loads(archive.read_text())
    print("Verified paired wrap, countdown, intervention, partial stores and word arithmetic.")
