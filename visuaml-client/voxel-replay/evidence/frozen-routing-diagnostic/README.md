# Frozen router diagnostic

This diagnostic uses the original tool-acquisition controller, graph, prompt, and tool images. One intervention replaces only the first result reported to the model. Two other interventions separately permute the textual registry and the physical tensor rows.

- Across feedback values 0 through 31, the model chose epsilon all 32 times. The stated rule calls for epsilon below 12 and zeta from 12 onward. It bound the supplied value on 27 of 32 requests.
- All six physical row orders produced the identical event trace when their identifiers, images, roles, and fuel moved together. Two of six textual orders produced the requested tool when the physical order was held constant.
- The saved logits come from the prefix immediately before the second tool name. At feedback 12, epsilon led zeta by 0.383 logits. The scalar reference gives 5 for zeta(15, 5).

These are interventions on one prompt. They isolate errors in the learned request without attributing those errors to a specific learned feature.

Run from `visuaml-client/voxel-replay`.

```sh
PYTHONPATH=.:../backend python tools/frozen_routing_diagnostic.py
```
