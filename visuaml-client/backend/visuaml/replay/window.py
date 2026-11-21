"""Reconstruct a bounded reference trace from a portable checkpoint."""
from .reports import REPORT_KIND, open_bundle

WINDOW_PROFILE = "visuaml-source-window-v1"
MAX_ROWS = 256


def inspect_window(document, *, after=None, limit=8, runner="reference"):
    """Inspect source transitions after an absolute retired-instruction boundary.

    The selected runner reaches the starting boundary. The reference graph
    reconstructs the requested rows. Trap and exhaustion rows can retire
    zero instructions. This validates the checkpoint, independent of any
    expected full-report result.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        raise ValueError(f"Trace length must be between 1 and {MAX_ROWS}")
    if document.get("report_kind") == REPORT_KIND:
        document = document["bundle"]
    replay, checkpoint = open_bundle(document)
    origin = document["retired_source_steps"]
    after = origin if after is None else after
    if type(after) is not int or after < origin:
        raise ValueError(f"Requested boundary precedes the saved source step {origin}")
    state = replay.restore(checkpoint)
    starting_fuel = int(state.vm.fuel)
    state = replay.run(state, after-origin, runner=runner)
    reached = origin+starting_fuel-int(state.vm.fuel)
    if reached != after:
        raise ValueError(f"Execution stopped after source step {reached}, before requested step {after}")
    at = dict(retired_source_steps=reached, pc=int(state.vm.pc), depth=int(state.depth),
              fuel=int(state.vm.fuel), status=int(state.vm.status),
              stack=state.vm.stack[:int(state.vm.height)].tolist())
    trace = []
    for _ in range(limit):
        if int(state.vm.status) != 0:
            break
        before = reached
        state, row = replay.step(state)
        reached = origin+starting_fuel-int(state.vm.fuel)
        trace.append(dict(row, retired_before=before, retired_after=reached))
    return dict(format_version=1, profile=WINDOW_PROFILE, checkpoint_sha256=checkpoint["checkpoint_sha256"],
                source_checkpoint_step=origin, skipped_source_steps=after-origin,
                seek_runner=runner if after > origin else None, trace_runner="reference",
                start=at, trace=trace, end=dict(retired_source_steps=reached, **replay.summary(state)))
