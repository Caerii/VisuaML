"""Author the post-freeze evaluation and independent expected outcomes."""
import json
from pathlib import Path

PROGRAMS = {
    "offset": [[1, 0, 0], [2, 0, 2], [1, 0, 2], [3, 0, 7], [2, 0, 0], [0, 0, 0]],
    "boost": [[1, 0, 0], [3, 0, 5], [3, 0, -3], [2, 0, 0], [3, 1, -1], [4, 1, 0], [0, 0, 0]],
    "trim": [[1, 0, 0], [3, 0, -2], [2, 0, 0], [3, 1, -1], [4, 1, 0], [0, 0, 0]],
}


def fixture(names, seed, count, expected, answer=None, *, order=(0, 1, 2),
            low="boost", high="trim", **kwargs):
    assert (3 * seed + count) % 7 == 0
    entries = list(zip(names, ("offset", "boost", "trim")))
    desc = " ".join(f"{name} does {family} ." for name, family in entries)
    prompt = (f"available {desc} please first offset {seed} once . "
              f"then choose {low} if below 12 otherwise {high} . {count} times . assistant")
    case = dict(prompt=prompt, registry=[entries[i] for i in order], calls=expected, **kwargs)
    if answer is not None:
        case["answer"] = answer
    return case


base = ("delta", "epsilon", "zeta")
renamed = ("gamma", "alpha", "beta")
initial = ["delta", 3, 1, 10]
original = fixture(base, 3, 5, [initial, ["epsilon", 10, 5, 20]], 20, order=(2, 0, 1))
cases = {
    "original": original,
    "substitution": fixture(base, 3, 5, [["delta", 3, 1, 15], ["zeta", 15, 5, 5]], 5,
                            order=(2, 0, 1), override=15),
    "renamed": fixture(renamed, 3, 5, [["gamma", 3, 1, 10], ["alpha", 10, 5, 20]], 20,
                       order=(1, 2, 0)),
    "shuffled_text": dict(original, prompt=original["prompt"].replace(
        "delta does offset . epsilon does boost . zeta does trim .",
        "epsilon does boost . zeta does trim . delta does offset ."), registry=[
            ["epsilon", "boost"], ["delta", "offset"], ["zeta", "trim"]]),
    "new_values": fixture(("epsilon", "beta", "delta"), 6, 3,
                          [["epsilon", 6, 1, 13], ["delta", 13, 3, 7]], 7, order=(1, 0, 2)),
    "reverse_intervention_exploratory": fixture(("epsilon", "beta", "delta"), 6, 3,
                                      [["epsilon", 6, 1, 8], ["beta", 8, 3, 14]], 14,
                                      order=(1, 0, 2), override=8),
    "swapped_descriptions": fixture(("delta", "zeta", "epsilon"), 3, 5,
                                   [initial, ["zeta", 10, 5, 20]], 20, order=(2, 0, 1)),
    "disabled": fixture(base, 3, 5, [["delta", 3, 1, "unavailable"]], fuel={"offset": -1}),
    "exhausted": fixture(base, 3, 5, [["delta", 3, 1, "exhausted"]], fuel={"offset": 2}),
    "invalid": fixture(renamed, 7, 0, [["gamma", 7, 1, 14], ["beta", 14, 0, "invalid"]]),
    "paraphrase_known_words": dict(original, prompt=original["prompt"].replace(
        "please first offset 3 once . then choose boost if below 12 otherwise trim . 5 times .",
        "kindly offset 3 once . if below 12 then boost 5 times otherwise trim 5 times .")),
    "paraphrase_free": dict(original, prompt=original["prompt"].split("please")[0] +
        "Take three and offset it. If the result is under twelve, boost it five times. "
        "If it is twelve or more, trim it five times. assistant"),
}
Path(__file__).with_name("holdout.json").write_text(
    json.dumps(dict(programs=PROGRAMS, cases=cases), indent=2) + "\n")
