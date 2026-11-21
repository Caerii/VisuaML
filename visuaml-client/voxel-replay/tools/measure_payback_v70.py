"""Paired lifecycle measurement for portable proof reuse across revision forks."""
from argparse import ArgumentParser
from dataclasses import asdict
import json
from random import Random
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from frozen_call_checkpoint import canonical, digest
from measure_uncertainty_v69 import deploy, engine, environment
from optimization_histories_v68 import OUT as V68, Authority, Branch, admit, binary
from persistent_evidence_v60 import Store
from verify_dependency_v65 import PAGE, SCRATCH

OUT = V68.parent / "proof-payback-v70"
MODE = "pointer"


def timed(fn):
    start = perf_counter_ns()
    value = fn()
    return value, perf_counter_ns() - start


def measure(runtime, baseline, candidate, n, calls, reverse):
    scores = {}
    variants = (("reload", baseline), ("pointer", candidate))
    for name, fn in reversed(variants) if reverse else variants:
        def run():
            for _ in range(calls):
                assert fn() == n
        _, scores[name] = timed(run)
    return scores


def trial(runtime, rng, calls, index):
    keys = rng.sample(range(1, 13), 8)
    changed = [next(k for k in range(1, 13) if k not in keys), *keys[1:]]
    page = rng.randbytes(PAGE)
    n = SCRATCH // (4 * len(keys))
    with TemporaryDirectory() as folder:
        library = Store(f"{folder}/shared")
        main = Branch(library)
        base = main.edit(keys)
        edited = main.fork()
        revised = edited.edit(changed)
        converged = edited.fork().edit(keys)
        assert len({base, revised, converged}) == 3
        shared = Authority(library)
        (root, raw, _), donor = timed(lambda: shared.prove(keys, MODE))

        def stale():
            try:
                admit(library, root, changed, MODE, shared.validation_cache)
            except ValueError as exc:
                return str(exc)
            raise AssertionError("stale certificate admitted")

        cold_order = ["cold", "shared", "edited_cold", "stale_repaired"]
        rng.shuffle(cold_order)
        cold_order.append("edited_shared")
        cases = {}
        for name in cold_order:
            layout = changed if name.startswith("edited") or name == "stale_repaired" else keys
            authority = shared
            if name in ("cold", "edited_cold"):
                authority = Authority(Store(f"{folder}/{name}"))
            reject_ns = 0
            if name == "stale_repaired":
                error, reject_ns = timed(stale)
            if name.endswith("shared"):
                identity = root if name == "shared" else cases["stale_repaired"]["root"]
                artifact, proof_ns = timed(lambda: admit(library, identity, layout, MODE,
                                                          shared.validation_cache))
                work = None
            else:
                (identity, artifact, work), proof_ns = timed(lambda: authority.prove(layout, MODE))
            candidate, deployment_ns = timed(lambda: deploy(runtime, artifact, page, layout, n))
            baseline = deploy(runtime, binary(layout, "reload"), page, layout, n)
            execution = measure(runtime, baseline, candidate, n, calls, (index + len(cases)) % 2)
            cases[name] = dict(root=identity, wasm=digest(artifact), work=work, proof_ns=proof_ns,
                               rejection_ns=reject_ns, deployment_ns=deployment_ns, execution_ns=execution,
                               net_ns=execution["reload"]-execution["pointer"]-proof_ns-reject_ns-deployment_ns)
        assert cases["cold"]["root"] == cases["shared"]["root"] == root
        assert cases["edited_cold"]["root"] == cases["edited_shared"]["root"] == cases["stale_repaired"]["root"]
        assert cases["stale_repaired"]["root"] != root
        assert cases["stale_repaired"]["work"]["fresh"] == 2
        assert raw == binary(keys, MODE)
        nodes = {path.name: json.loads(path.read_bytes()) for path in library.root.iterdir()}
        return dict(keys=keys, changed=changed, revisions=[base, revised, converged],
                    page=digest(page), records_per_call=n, donor_ns=donor, rejection=error,
                    case_order=cold_order, cases=cases), nodes


def replay(samples, nodes):
    rng = Random(7002)
    runtime = engine("none")
    with TemporaryDirectory() as folder:
        store = Store(folder)
        for name, value in nodes.items():
            (store.root / name).write_bytes(canonical(value))
        for sample in samples:
            assert rng.sample(range(1, 13), 8) == sample["keys"]
            page = rng.randbytes(PAGE)
            assert digest(page) == sample["page"]
            order = ["cold", "shared", "edited_cold", "stale_repaired"]
            rng.shuffle(order)
            assert order + ["edited_shared"] == sample["case_order"]
            base, edited, converged = (store.load(item) for item in sample["revisions"])
            assert base.parents == () and base.payload["keys"] == sample["keys"]
            assert edited.parents == (sample["revisions"][0],) and edited.payload["keys"] == sample["changed"]
            assert converged.parents == (sample["revisions"][1],) and converged.payload["keys"] == sample["keys"]
            for name, item in sample["cases"].items():
                keys = sample["changed"] if name.startswith("edited") or name == "stale_repaired" else sample["keys"]
                artifact = admit(store, item["root"], keys, MODE, set())
                assert digest(artifact) == item["wasm"]
                for raw in (binary(keys, "reload"), artifact):
                    deploy(runtime, raw, page, keys, sample["records_per_call"])
            try:
                admit(store, sample["cases"]["shared"]["root"], sample["changed"], MODE, set())
            except ValueError:
                continue
            raise AssertionError("stale certificate admitted on replay")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    results = OUT / "samples.json"
    proofs = OUT / "proofs.json"
    if not args.measure:
        replay(json.loads(results.read_bytes())["samples"], json.loads(proofs.read_bytes()))
        print("Rechecked held-out certificates and stale rejection with fresh verifier caches")
        return
    runtime = engine("none")
    pilot_rng = Random(7001)
    pilots = [trial(runtime, pilot_rng, 2048, i)[0] for i in range(3)]
    thresholds = []
    for sample in pilots:
        case = sample["cases"]["cold"]
        execution = case["execution_ns"]
        gain = (execution["reload"] - execution["pointer"]) / 2048
        if gain > 0:
            shared = sample["cases"]["shared"]
            lower = shared["proof_ns"] + shared["deployment_ns"]
            upper = case["proof_ns"] + case["deployment_ns"]
            thresholds.append((lower + upper) / (2 * gain))
    calls = min(16000, max(256, round(median(thresholds)))) if thresholds else 2048
    rng = Random(7002)
    samples, nodes = [], {}
    for index in range(20):
        sample, records = trial(runtime, rng, calls, index)
        samples.append(sample)
        nodes.update(records)
    results.write_bytes(canonical(dict(environment=asdict(environment("none")), calls=calls,
                                       pilots=pilots, pilot_calls=2048, samples=samples))+b"\n")
    proofs.write_bytes(canonical(nodes)+b"\n")
    print(f"{len(samples)} paired trials at {calls} invocations per variant")


if __name__ == "__main__":
    main()
