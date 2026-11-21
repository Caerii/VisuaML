# Replay a model intervention with exact WASM feedback

This example uses two archived width-32 routing checkpoints. It keeps
the weights fixed. An unchanged Rust WASM guest computes the first
feedback value. VisuaML capture independently checks its return and full
page against Wasmtime. The harness passes the result to the controller
and observes its proposed next registry slot. The first tool is fixed
by the fixture. The controller also emits a first-choice score as a
control. The experiment stops at the second request.

From `visuaml-client/voxel-replay` in the research environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_activation_intervention.py
python tools/visuaml_replay.py open release/examples/interventions/tool-report.json
```

Load [the experiment JSON](../examples/interventions/experiment.json) in
the replay viewer to see the selected case. The tool returned 6. A patch
to `router.condition.1` switched the next registry slot from 0 to 1
while the controller still received 6. In the feedback substitution,
the tool again returned 6 and the host supplied 7. The first observable
difference in the donor patch is the second request. For the feedback
substitution it is the feedback boundary.

## Controls and scope

| Intervention | Second-choice switches out of 128 eligible pairs |
| --- | ---: |
| Opposite-side donor | 128 |
| Opposite-side feedback | 128 |
| Same-side donor | 0 |
| Norm-matched random direction | 7 |
| No-op | 0 |
| Zero ablation | 64 |

Eligibility requires both unpatched endpoints to choose correctly; all
128 qualified. Task labels define the expected switch before the patch.
Every intervention preserved the first choice. Zero ablation shows what
happened under a coarse intervention; it has no predicted winner.
Random patches use one fixed direction sample per case, with the same
perturbation norm as the paired donor.

The hook observes the output of the tanh module in the learned condition
network. The baseline was checked with and without the hook. At that site,
the patched second scores exactly matched the live opposite-side feedback
scores. The checkpoints and threshold cases come from the earlier
routing research. This replay adds real WASM feedback and rotates registry
slots without retraining. The two registry variants are correlated.
This is a small CPU controller and a full activation substitution.
The result establishes sufficiency at the selected site for these cases.
It does not isolate a neuron or establish a mechanism in a language model.

## Use the hook in another experiment

```python
from visuaml.replay.interventions import ActivationSite, run_intervention

site = ActivationSite("router.condition.1")
outputs, original, delivered = run_intervention(model, site, inputs, replacement=donor)
```

A site names one PyTorch module output during a forward pass. The hook
requires a single invocation and a tensor with matching shape, dtype and
device. It returns detached snapshots and removes the hook on completion
or error. Use an isolated model instance for concurrent requests. This
adapter supports eager PyTorch module outputs. Prefill, token and KV-cache
sites need adapters with their own semantics.

Case inputs, frozen weight hashes, the tool hash, output logits, activation
digests and predictions are in the report. It stores digests instead of
full activation arrays. The reproducer reruns the intervention and checks
the archived observations. [CPU timing samples](../examples/interventions/timings.json)
cover the batched cases and report assembly, excluding the WASM setup
and artifact I/O.
