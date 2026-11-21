"""Separate composition, token distance, and checkpoint controls for the frozen decoder."""
import argparse
import json
from random import Random
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from frozen_call_checkpoint import digest
from replay_decoder_execution_loop import run
from replay_word_decoder import audit, pack
from word_decoder import OUT as WORD, COMMIT, FRAME, HALT, NONE, PAD

OUT = WORD.parent / "word-research"


def compose(model):
    rng = Random(41211)
    results = []
    # Template A inserts independent stores/loads/arithmetic in noncanonical order.
    # Template B uses a forward branch across such a block.
    for case in range(40):
        block = [[op, 0, rng.randrange(4) if op != 3 else rng.choice((-1, 1, 3, 2147483647))]
                 for op in rng.sample([1, 2, 3], 3)]
        program = [[1,0,0]] + block + [[2,0,3], [1,0,3]]
        if case % 2:
            program.insert(1, [4,0,5])
        program += [[0,0,0]]
        memory = [rng.choice((0, 1, 0xffffffff, 0xfffffffe, rng.getrandbits(32))) for _ in range(4)]
        init = pack(program, memory)
        tokens, _, _, _ = run(model, init, True, budget=128, halt_token=HALT)
        states, _ = audit(tokens, memory)
        expected = reference(program, memory, 0, 16)
        if states != expected:
            boundary = next(i for i in range(max(len(states), len(expected)))
                            if states[i:i+1] != expected[i:i+1])
            raise AssertionError((case, boundary+1, tokens[boundary*FRAME:(boundary+1)*FRAME]))
        results.append(dict(program=program, memory=memory, tokens=tokens, states=states))
    return results


def context(model):
    program = [[1,0,0],[4,0,3],[0,0,0],[0,0,0]]
    memory = [1, 0, 0, 0]
    base = pack(program, memory)
    reference_state = reference(program, memory, 0, 3)
    out = []
    for depth in (0, 8, 20, 29):
        for content in ("padding", "distractor"):
            additional = []
            for index in range(depth):
                if content == "padding":
                    additional.extend([PAD]*FRAME)
                elif index % 3 == 0:
                    # Same-address all-ones payload with no COMMIT.
                    additional.extend([PAD,0,0,0,0,0,0,255,255,255,255,PAD])
                else:
                    # Interleaved no-effect writes to other addresses.
                    additional.extend([PAD,0,0,0,0,0,1+index%3,0,0,0,0,COMMIT])
            # Put an older competing COMMIT before the authoritative initialization.
            old = torch.tensor([PAD,0,0,0,0,0,0,255,255,255,255,COMMIT])
            init = torch.cat((base[:4*FRAME], old, base[4*FRAME:-FRAME],
                              torch.tensor(additional, dtype=torch.long), base[-FRAME:]))
            times = []
            stream = None
            run(model, init, True, budget=36, halt_token=HALT)
            for _ in range(5):
                tokens, prefill, decode, cache = run(model, init, True, budget=36, halt_token=HALT)
                assert stream is None or stream == tokens
                stream = tokens
                times.append(dict(prefill_ns=prefill, decode_ns=decode))
            assert audit(stream, memory)[0] == reference_state
            assert stream == run(model, init, False, budget=36, halt_token=HALT)[0]
            # Query at first generated EXEC, with the known address 0.
            keys = cache[2][0][0, :len(init)+1]
            scores = 8192*keys[:,0]+keys[:,-1]
            best = scores.topk(2)
            assert int(best.indices[0]) == 6*FRAME-1
            out.append(dict(padding_frames=depth, content=content,
                            read_distance=len(init)-int(best.indices[0]),
                            score_margin=int(best.values[0]-best.values[1]), prefix_tokens=len(init),
                            generated_tokens=len(stream), stream=stream,
                            cache_bytes=sum(t.numel()*t.element_size() for pair in cache for t in pair),
                            timings=times))
    return out


def continuation(model):
    program = [[1,0,0],[3,0,-1],[2,0,0],[4,0,0],[0,0,0]]
    memory = [3,0,0,0]
    initial = pack(program, memory)
    complete, _, _, _ = run(model, initial, True, budget=192, halt_token=HALT)
    states, _ = audit(complete, memory)
    # Restore after both a store and branch, at a completed source instruction.
    cut = 4
    saved = states[cut-1]
    restored = pack(program, saved[4:])
    restored[-FRAME+1] = saved[0]
    restored[-FRAME+2:-FRAME+6] = torch.tensor(list(saved[2].to_bytes(4, "little")))
    continued, _, _, _ = run(model, restored, True, budget=192, halt_token=HALT)
    assert continued == complete[cut*FRAME:]
    return dict(checkpoint_step=cut, state=saved, removed_history_tokens=cut*FRAME,
                suffix_tokens=continued, remaining_steps=len(continued)//FRAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = WORD / "decoder.pt"
    model = torch.jit.load(str(path))
    with torch.inference_mode():
        record = dict(decoder_sha256=digest(path.read_bytes()), compositions=compose(model),
                      context=context(model), continuation=continuation(model))
    archive = OUT / "execution.json"
    if args.write:
        archive.write_text(json.dumps(record, indent=2)+"\n")
    else:
        saved = json.loads(archive.read_text())
        for sweep in (saved, record):
            for row in sweep["context"]:
                del row["timings"]
        assert saved == record
    print("Verified 40 generated compositions, eight context controls and checkpoint continuation.")
