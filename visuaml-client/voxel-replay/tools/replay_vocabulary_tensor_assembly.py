"""Compare emitted tokens with the scalar, recurrent and structured references."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from audit_causal_assembly import reference
from causal_tensor_assembly import decode, observe, start
from frozen_call_checkpoint import digest
from internal_tool import prepare_program
from synthesize_attention_vm import OUT as RECURRENT
from vocabulary_tensor_assembly import OUT, STRUCTURED, FRAME, CONTEXT, COMMIT, HALT, encode_image, generate


def replay(measure):
    vocab_path = OUT / "decoder.pt"
    structured_path = STRUCTURED / "decoder.pt"
    recurrent_path = RECURRENT / "executor.pt"
    decoder, structured, recurrent = [torch.jit.load(str(path))
                                      for path in (vocab_path, structured_path, recurrent_path)]
    fixtures = json.loads((STRUCTURED / "holdout.json").read_text())
    holdout = json.loads((OUT / "holdout.json").read_text())
    results, timings = [], {}
    for case in fixtures + holdout:
        code, seed, count = case["program"], case["seed"], case["counter"]
        memory = [seed, 0, 0, 0]
        image = start(code, memory, count)
        initial = encode_image(image)
        fuel = min(case["fuel"], (CONTEXT-len(initial)) // FRAME)
        tokens = generate(decoder, initial, fuel * FRAME)
        records = decoder.committed(tokens)
        expected = reference(code, memory, count, fuel)
        actual = observe(records, memory)
        _, recurrent_trace = recurrent(prepare_program(code), torch.tensor([seed, count]))
        structured_records = decode(structured, image, fuel)
        for name, other in (("scalar", expected), ("recurrent", recurrent_trace[:len(expected)].tolist()),
                            ("structured", observe(structured_records, memory))):
            first = next((i for i in range(max(len(other), len(actual)))
                          if actual[i:i+1] != other[i:i+1]), None)
            if first is not None:
                raise AssertionError(dict(case=case["name"], backend=name, boundary=first+1,
                                          tokens=tokens[first*FRAME:(first+1)*FRAME].tolist(),
                                          actual=actual[first:first+1], expected=other[first:first+1]))
        assert torch.equal(records, structured_records)
        assert len(tokens) == len(records)*FRAME
        assert (int(tokens[-1]) == HALT) == (actual[-1][1] == 0)
        assert all(int(marker) == COMMIT for marker in tokens[FRAME-1:-1:FRAME])
        check = dict(name=case["name"], initial_tokens=initial.tolist(), emitted_tokens=tokens.tolist(),
                     trace=actual, source_steps=len(actual))
        if case is holdout[0]:
            before_store, after_store = 3*FRAME, 4*FRAME
            before = torch.cat((initial, tokens[:before_store]))
            unchanged = decoder.committed(before)
            for phase in range(1, FRAME):
                partial = torch.cat((before, tokens[before_store:before_store+phase]))
                assert torch.equal(decoder.committed(partial), unchanged)
                poisoned = torch.cat((before, (tokens[before_store:before_store+phase]+127) % 256))
                assert torch.equal(decoder(partial), decoder(poisoned))
                assert int(decoder(partial).argmax()) == int(tokens[before_store+phase])
            markerless = torch.cat((initial, tokens[:after_store]))
            markerless[-1] = 0
            try:
                decoder.committed(markerless)
            except torch.jit.Error:
                pass
            else:
                raise AssertionError("A full payload without a commit marker was accepted")
            # Offline intervention: replace just the committed write payload. The following
            # generated LOAD should now return the same value as the older store.
            altered = records[3:4].clone()
            altered[:, 27:59] = records[1:2, 27:59]
            prefix = torch.cat((before, encode_image(altered)))
            replacement_load = generate(decoder, prefix, FRAME)
            replacement_state = observe(decoder.committed(replacement_load), memory)[0]
            assert actual[4][3] == 4294967294 and replacement_state[3] == 4294967295
            check["write_audit"] = dict(store_tokens=[before_store, after_store], load_tokens=[after_store, 5*FRAME],
                                        all_partial_phases_invisible=20, committed_read=actual[4][3],
                                        replaced_write_read=replacement_state[3],
                                        replacement_load_tokens=replacement_load.tolist())
        results.append(check)
        if measure:
            samples = {"token_ns": [], "structured_ns": []}
            for index in range(12):
                phases = [("token_ns", lambda: generate(decoder, initial, fuel*FRAME)),
                          ("structured_ns", lambda: decode(structured, image, fuel))]
                for key, call in phases[::(-1 if index % 2 else 1)]:
                    began = perf_counter_ns()
                    call()
                    samples[key].append(perf_counter_ns()-began)
            timings[case["name"]] = dict(samples=samples, source_steps=len(actual),
                                         tokens_per_step=FRAME, final_context_tokens=len(initial)+len(tokens),
                                         token_storage_bytes=(len(initial)+len(tokens))*initial.element_size(),
                                         persistent_kv_bytes=0)
    return dict(profile="visuaml-vocabulary-tensor-assembly-v1", holdout=holdout,
                decoder_sha256=digest(vocab_path.read_bytes()), structured_sha256=digest(structured_path.read_bytes()),
                recurrent_sha256=digest(recurrent_path.read_bytes()), first_divergence=None, cases=results), timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--open", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        bundle, samples = replay(args.write)
    if args.write:
        for name, item in (("replay", bundle), ("timings", samples)):
            (OUT / (name+".json")).write_text(json.dumps(item, indent=2)+"\n")
    else:
        assert bundle == json.loads((args.open or OUT / "replay.json").read_text())
    print(f"Matched {len(bundle['cases'])} cases and "
          f"{sum(case['source_steps'] for case in bundle['cases'])} boundaries.")
