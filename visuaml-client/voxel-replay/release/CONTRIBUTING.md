# Contribute to VisuaML Replay

Install the package in a virtual environment using the release quickstart.
Run the tests from `visuaml-client/voxel-replay`.

```sh
export PYTHONPATH=../backend
python tools/test_runnable_report.py
python tools/test_replay_window.py
python -m doctest release/chapters/suspended-helper.md
```

The public API is in `backend/visuaml/replay`. Keep guest admission and
execution semantics in the runtime, portable state and recipes in the
checkpoint and report modules, and argument parsing in the CLI. The viewer
presents saved JSON. Use the unmodified report from a prior release when
testing compatibility.

For a bug, include the producing version, report or minimal guest, command,
expected observation, and first actual difference. Use a synthetic memory
page when sharing publicly. For a semantic change, add a reproducible case
that fails on the old code. Name the observation boundary the test checks.
