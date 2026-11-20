"""Run in a separate environment with pinned upstream on PYTHONPATH."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import torch
from transformer_vm.attention import StandardKVCache
from transformer_vm.model.weights import load_weights
from transformer_vm.specialize import specialize


def baseline(model, vocab, indices, tokens, expected):
    prefix = torch.tensor([[indices[x] for x in tokens]])
    samples = []
    mismatch = None
    with torch.inference_mode():
        for _ in range(3):
            start = perf_counter_ns()
            generated = model.generate_with_cache(prefix, max_new_tokens=1000, cache_class=StandardKVCache)
            samples.append(perf_counter_ns()-start)
            actual = [vocab[i] for i in generated[0].tolist()]
            if actual != expected:
                bad = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                           min(len(actual), len(expected)))
                mismatch = dict(index=bad, actual=actual[bad:bad+4], expected=expected[bad:bad+4],
                                actual_tokens=len(actual), actual_tail=actual[-8:])
                break
    return dict(prefix_tokens=len(tokens), full_reference_tokens=len(expected), execution_ns=samples,
                terminal_token=expected[-1], first_difference=mismatch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("prefix", type=Path, help="Path without .txt extension")
    args = parser.parse_args()
    torch.set_num_threads(1)
    program = str(args.prefix)+".txt"
    tokens = Path(program).read_text().split()
    reference = Path(str(args.prefix)+"_ref.txt").read_text().split()
    specialized = Path(str(args.prefix)+"_spec.txt").read_text().split()
    assert reference[:len(tokens)] == tokens
    start = perf_counter_ns()
    universal_model = load_weights(str(args.weights))
    load = perf_counter_ns()-start
    universal = baseline(*universal_model, tokens, reference)
    assert universal["first_difference"] is None
    start = perf_counter_ns()
    specialized_model = specialize(program)[:3]
    prepare = perf_counter_ns()-start
    baked = baseline(*specialized_model, specialized, specialized + reference[len(tokens):])
    print(json.dumps(dict(universal=universal, specialized=baked,
                          universal_load_ns=load, specialized_build_ns=prepare,
                          universal_weights_sha256=sha256(args.weights.read_bytes()).hexdigest(),
                          torch=str(torch.__version__)), indent=2))


if __name__ == "__main__":
    main()
