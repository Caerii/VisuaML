"""Versioned checkpoint bundles and reproducible report recipes."""
import base64

import wasmtime

from .checkpoint import canonical, digest
from .divergence import first_divergence
from .runtime import Replay

PROFILE = "visuaml-direct-call-i32-v1"
REPORT_KIND = "visuaml-runnable-report"


def make_bundle(replay, argument, initial, fuel, count, *, runner="reference"):
    if not 0 <= count < fuel:
        raise ValueError("Checkpoint step must be within the initial fuel allowance")
    state = replay.start(argument, initial, fuel)
    state = replay.run(state, count, runner=runner)
    if int(state.vm.status) != 0:
        raise ValueError("Requested checkpoint is terminal")
    bundle = dict(format_version=1, profile=PROFILE, wasm=base64.b64encode(replay.raw).decode(),
                  retired_source_steps=count, checkpoint=replay.checkpoint(state))
    return dict(bundle, bundle_sha256=digest(canonical(bundle)))


def open_bundle(bundle):
    if bundle.get("format_version") != 1 or bundle.get("profile") != PROFILE:
        raise ValueError("Unsupported replay bundle")
    if type(bundle.get("retired_source_steps")) is not int or bundle["retired_source_steps"] < 0:
        raise ValueError("Invalid retired source step count")
    payload = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if digest(canonical(payload)) != bundle.get("bundle_sha256"):
        raise ValueError("Bundle integrity mismatch")
    raw = base64.b64decode(bundle["wasm"], validate=True)
    return Replay(raw), bundle["checkpoint"]


def compare(bundle, preview_local):
    replay, checkpoint = open_bundle(bundle)
    baseline = replay.finish(checkpoint)
    candidate = replay.finish(checkpoint, preview_local)
    boundary = first_divergence(baseline["trace"], candidate["trace"])
    first = None if boundary is None else dict(
        retired_source_steps=bundle["retired_source_steps"]+boundary+1,
        reference=baseline["trace"][boundary], candidate=candidate["trace"][boundary])
    return dict(format_version=1, profile=PROFILE, checkpoint_sha256=checkpoint["checkpoint_sha256"],
                preview_local=preview_local, first_difference=first, reference=baseline, candidate=candidate)


def execute_report(bundle, recipe):
    if not isinstance(bundle, dict) or not isinstance(recipe, dict):
        raise ValueError("Expected a checkpoint bundle and a report recipe")
    if recipe.get("kind") == "expect-return":
        expected = recipe.get("value")
        if set(recipe) != {"kind", "value"} or type(expected) is not int or not -2**31 <= expected < 2**32:
            raise ValueError("Expected return must be an i32 value")
        replay, checkpoint = open_bundle(bundle)
        result = replay.finish(checkpoint)
        return dict(result, expected_return=expected, expectation_met=(
            result["status"] == 1 and result["returned"] == expected & 0xffffffff))
    if recipe == {"kind": "resume"}:
        replay, checkpoint = open_bundle(bundle)
        return replay.finish(checkpoint)
    if set(recipe) == {"kind", "local"} and recipe["kind"] == "local.get-preview":
        if type(recipe["local"]) is not int:
            raise ValueError("Preview local must be an integer")
        return compare(bundle, recipe["local"])
    raise ValueError("Unsupported report recipe")


def seal_report(bundle, recipe):
    result = execute_report(bundle, recipe)
    return dict(report_kind=REPORT_KIND, report_version=1, bundle=bundle, recipe=recipe,
                expected_result_sha256=digest(canonical(result)))


def check_report(report):
    if type(report.get("report_version")) is not int or report["report_version"] != 1:
        raise ValueError("Unsupported runnable report version")
    result = execute_report(report["bundle"], report["recipe"])
    if digest(canonical(result)) != report["expected_result_sha256"]:
        raise ValueError("Runnable report failed to reproduce the recorded result")
    return result

def capture(replay, argument, initial, fuel, count, *, runner="reference"):
    """Capture the requested boundary and verify successful completion independently."""
    bundle = make_bundle(replay, argument, initial, fuel, count, runner=runner)
    completed = replay.finish(bundle["checkpoint"])
    if completed["status"] == 1:
        store = wasmtime.Store()
        instance = wasmtime.Instance(store, wasmtime.Module(store.engine, replay.raw), [])
        exports = instance.exports(store)
        exports["memory"].write(store, initial)
        result = exports["run"](store, argument)
        page = exports["memory"].read(store)
        if result & 0xffffffff != completed["returned"] or digest(page) != completed["page_sha256"]:
            raise ValueError("Source-step runtime and Wasmtime disagree")
    return bundle
