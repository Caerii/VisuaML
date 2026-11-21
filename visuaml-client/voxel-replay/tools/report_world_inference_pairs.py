"""Render family-level decisions and package examples from the locked evaluation."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import torch

from evaluate_world_inference_pairs import (FAMILIES,HOLDOUT_SEED,OUT,FROZEN,
                                             arrival,build_pairs,run)
from frozen_call_checkpoint import canonical


def plot(data):
    width,height = 800,292
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
           '<title id="title">Second-call selection by feedback value and layout family</title>',
           '<rect width="800" height="292" rx="10" fill="#f2f4eb"/>',
           '<g font-family="system-ui,sans-serif" fill="#243831">',
           '<text x="24" y="32" font-size="19" font-weight="700">The second call follows the feedback bit</text>',
           '<text x="24" y="54" font-size="12">Fraction selecting the tool designated for bit 1 · 25 world pairs per panel</text>']
    for index,family in enumerate(FAMILIES):
        x = 53+194*index
        svg.extend([f'<text x="{x+49}" y="84" text-anchor="middle" font-size="13">{escape(family)}</text>',
                    f'<path d="M{x} 101 V219 H{x+98}" stroke="#92a299" fill="none"/>'])
        if index==0:
            svg.extend(['<text x="29" y="107" font-size="11">1.0</text>','<text x="29" y="223" font-size="11">0.0</text>'])
        rows = [r for r in data if r["family"]==family]
        for mode,color,dash in (("actual","#264b42",""),("swapped","#c36023",' stroke-dasharray="4 4"')):
            points=[]
            for bit in (0,1):
                values=[]
                for row in rows:
                    high = row["runs"]["actual"]["1"]["prediction"]["tool"]
                    for run_record in row["runs"][mode].values():
                        if run_record["prediction"]["reported"]==bit:
                            values.append(run_record["requests"][1][1:2]==[high])
                points.append((x+13+bit*72,219-118*sum(values)/len(values)))
            svg.append(f'<path d="M{points[0][0]} {points[0][1]} L{points[1][0]} {points[1][1]}" stroke="{color}" stroke-width="3" fill="none"{dash}/>')
            for px,py in points:
                svg.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{color}"/>')
        svg.append(f'<text x="{x+13}" y="239" font-size="11">0</text><text x="{x+85}" y="239" font-size="11">1</text>')
    svg += ['<text x="24" y="269" font-size="12">Feedback seen</text>',
            '<path d="M249 265 h25" stroke="#264b42" stroke-width="3"/><text x="280" y="269" font-size="12">Actual</text>',
            '<path d="M354 265 h25" stroke="#c36023" stroke-width="3" stroke-dasharray="4 4"/><text x="385" y="269" font-size="12">Substituted</text>',
            '<text x="504" y="269" font-size="12">Expected switch: bit 0 → 0%, bit 1 → 100%</text>',
            '</g></svg>']
    (OUT/"decisions.svg").write_text("\n".join(svg)+"\n")


def examples(archive,pairs):
    records = archive["pairs"]
    matching = {p["id"]:p for p in pairs}
    switch = next(row for row in records if all(r["decision_correct"] for mode in
                  ("actual","swapped") for r in row["runs"][mode].values()))
    stability = next(row for row in records if row["family"]=="elbow" and all(
                     r["decision_correct"] for r in row["runs"]["irrelevant"].values()))
    failure = next((dict(record=row,mode=mode,bit=bit) for mode in ("actual","swapped","free")
        for row in records for bit,details in row["runs"][mode].items() if details["first_wrong"]),None)
    assert failure is not None
    targets = dict(switch=dict(fixture=matching[switch["id"]],record=switch),
                   irrelevant=dict(fixture=matching[stability["id"]],record=stability),
                   failure=dict(fixture=matching[failure["record"]["id"]],**failure))
    for name,entry in targets.items():
        (OUT/(name+".json")).write_bytes(canonical(entry)+b"\n")
    return targets


def timing_boundary(pairs):
    # Supplemental evaluation holds each late world fixed and changes just its horizon.
    rows = []
    torch.set_num_threads(1)
    model = torch.jit.load(str(FROZEN/"inference.pt"))
    with torch.inference_mode():
        for pair in pairs:
            late = pair["worlds"][0]
            threshold = arrival(late)
            if pair["family"]=="blocked-ingress" or threshold>8:
                continue
            short = run(model,dict(pair,horizon=threshold-1,worlds=[late,late]),0,"actual")
            long = run(model,dict(pair,horizon=threshold,worlds=[late,late]),1,"actual")
            rows.append(dict(pair=pair["id"],arrival=threshold,short=short,long=long))
    return dict(scope="supplemental fixed-geometry horizon control",cases=rows,
                correct_switch=sum(r["short"]["decision_correct"] and r["long"]["decision_correct"] for r in rows))


if __name__ == "__main__":
    archive = json.loads((OUT/"outcomes.json").read_text())
    plot(archive["pairs"])
    pairs = build_pairs(HOLDOUT_SEED,25)
    examples(archive,pairs)
    control = timing_boundary(pairs)
    (OUT/"horizon-control.json").write_bytes(canonical(control)+b"\n")
    print(f"Fixed geometry horizon switches: {control['correct_switch']}/{len(control['cases'])}.")
