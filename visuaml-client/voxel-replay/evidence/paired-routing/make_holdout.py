"""Define inquiry cases and semantic outcomes after freezing the router."""
import json
from pathlib import Path

PROGRAMS = {
    "offset": [[1, 0, 0], [3, 0, 7], [2, 0, 0], [0, 0, 0]],
    "boost": [[1, 0, 0], [3, 0, 5], [2, 0, 2], [1, 0, 2], [3, 0, -2],
              [2, 0, 0], [3, 1, -1], [4, 1, 0], [0, 0, 0]],
    "trim": [[1, 0, 0], [3, 0, -3], [3, 0, 1], [2, 0, 0], [3, 1, -1], [4, 1, 0], [0, 0, 0]],
}
FAMILIES = ("offset", "boost", "trim")
BASE = (1441, 70, 860)


def expect(offset, boost, trim, seed, repeat, threshold, relation, override=None,
           choices=(1, 2), forced=None, unavailable=False, exhausted=False):
    first = seed + 7
    if unavailable or exhausted:
        status = 2 if unavailable else 3
        return [[offset, seed, 1, status, -1, -1]]
    feedback = override if override is not None else first
    takes_first = (feedback < threshold) if relation == 0 else (feedback >= threshold)
    target = choices[0 if takes_first else 1]
    handle = [offset, boost, trim][target]
    if forced is not None:
        handle = forced
    if handle not in (boost, trim) or not 1 <= repeat <= 5:
        second = [handle, feedback, repeat, 1, -1, -1]
    else:
        answer = (feedback + (3 if handle == boost else -2) * repeat) % 2**32
        second = [handle, feedback, repeat, int(answer > 31), answer, answer]
    return [[offset, seed, 1, 0, first, feedback], second]


def fixture(handles=BASE, seed=3, repeat=2, threshold=12, relation=0, override=None,
            order=(0, 1, 2), unseen=False, choices=(1, 2), forced=None,
            unavailable=False, exhausted=False):
    phrases = (("shift the seed value", "raise the input value", "reduce the input value")
               if unseen else ("offset the value", "raise each repeat", "repeat a trim"))
    queries = (("offset the value", "boost the input", "lower the input")
               if unseen else ("offset the value", "increase value", "lower each repeat"))
    records = [{"handle": handles[i], "label": ["Ibis", "Orbit", "Saffron"][i],
                "family": FAMILIES[i], "description": phrases[i],
                "schema": [0, 31, 1, 1 if i == 0 else 5],
                "fuel": (-1 if unavailable else 2 if exhausted else 48) if i == 0 else 48}
               for i in order]
    return dict(category="new_description_compositions" if unseen else "familiar_descriptions_new_programs",
                registry=records, choices=[queries[i] for i in choices], seed=seed, repeat=repeat,
                threshold=threshold, relation=relation, override=override, forced_handle=forced,
                expected=expect(*handles, seed, repeat, threshold, relation, override, choices,
                                forced, unavailable, exhausted))


cases = {
    "original": fixture(),
    "substitute_at_boundary": fixture(override=12),
    "reorder": fixture(order=(2, 0, 1)),
    "fresh_handles": fixture(handles=(987, 12112, 604), override=12, order=(2, 1, 0)),
    "reverse_relation": fixture(override=12, relation=1),
    "new_threshold_and_choice_order": fixture(seed=9, threshold=15, choices=(2, 1)),
    "new_description_compositions": fixture(unseen=True, seed=4),
    "new_description_switch": fixture(unseen=True, seed=4, override=13, order=(1, 2, 0)),
    "invalid_argument": fixture(override=12, repeat=0),
    "disabled": fixture(unavailable=True),
    "exhausted": fixture(exhausted=True),
    "unknown_handle_probe": fixture(forced=641),
    "forced_correct_handle_probe": fixture(override=12, forced=860),
}
# This worksheet is intentionally outside the structured interface.
cases["free_paraphrase"] = dict(category="free_paraphrase",
                                prompt="Give the seed some lift. If it comes in under a dozen, "
                                       "ramp it up twice; otherwise dial it down twice.",
                                expected_admission=False)
sweep = {str(value): fixture(override=value) for value in range(32)}
Path(__file__).with_name("holdout.json").write_text(json.dumps(
    dict(programs=PROGRAMS, cases=cases, sweep=sweep), indent=2) + "\n")
