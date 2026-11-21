"""Summarize held-out v70 lifecycle results with paired layout resampling."""
import json
from random import Random
from statistics import median, quantiles

from measure_payback_v70 import OUT


def summary(values):
    rng = Random(7003)
    intervals = quantiles(
        [median(rng.choices(values, k=len(values))) for _ in range(10000)], n=40)
    return dict(median_ms=median(values)/1e6, ci95_ms=[intervals[0]/1e6, intervals[-1]/1e6],
                positive=sum(value > 0 for value in values))


def main():
    report = json.loads((OUT / "samples.json").read_bytes())
    samples = report["samples"]
    rows = [sample["cases"] for sample in samples]
    names = ("cold", "shared", "edited_cold", "stale_repaired", "edited_shared")
    result = {
        name: dict(
            net=summary([row[name]["net_ns"] for row in rows]),
            gross=summary([row[name]["execution_ns"]["reload"]-row[name]["execution_ns"]["pointer"]
                           for row in rows]),
            proof_ms=median(row[name]["proof_ns"] for row in rows)/1e6,
            rejection_ms=median(row[name]["rejection_ns"] for row in rows)/1e6,
            deployment_ms=median(row[name]["deployment_ns"] for row in rows)/1e6,
        )
        for name in names
    }
    result["contrasts"] = {
        "shared_minus_cold": summary([r["shared"]["net_ns"]-r["cold"]["net_ns"] for r in rows]),
        "stale_minus_edited_cold": summary(
            [r["stale_repaired"]["net_ns"]-r["edited_cold"]["net_ns"] for r in rows]),
        "shared_charged_donor": summary(
            [s["cases"]["shared"]["net_ns"]-s["donor_ns"] for s in samples]),
        "same_trial_sign_flip": sum(r["cold"]["net_ns"] <= 0 < r["shared"]["net_ns"] for r in rows),
        "donor_ms": median(s["donor_ns"] for s in samples)/1e6,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
