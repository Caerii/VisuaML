# Fresh-process release handoff

This internal holdout was authored after freezing release candidate `803772e`. Its capture, resume and compare ran in independent CLI processes. The checkpoint is inside the helper at depth 1. The original returns 45. The first proposed preview exposes 33 where the reference exposes 12. The completed outcomes agree.

```sh
export PYTHONPATH=.:../backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/visuaml_replay.py capture evidence/replay-rc-handoff/holdout.wasm /tmp/holdout.json --argument 12 --after-step 4
python tools/visuaml_replay.py open /tmp/holdout.json
python tools/visuaml_replay.py compare /tmp/holdout.json --preview-local 1
```

The outside handoff remains a separate usability test with a contributor and recipient using their own program.
