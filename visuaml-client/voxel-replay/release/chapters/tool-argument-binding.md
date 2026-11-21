# Separate the tool choice from its argument slot

The first guest accepts two packed bytes, increments each, and stores and
returns the result. Each replay uses adjacent values in the range where
adding 257 increments both bytes without carry between them. The host
unpacks the bytes and supplies the tagged live value to the frozen seed
829 router. The binding head receives routing logits, the slot layout,
and the requested role (live or saved). It emits a physical slot. The
host resolves that slot to its recorded value and dispatches Boost, Trim,
or no call. This tests learned selection between explicit slots. The
head does not generate an i32 or infer which role a slot represents
without layout metadata.

The binding head is trained at thresholds 9 and 17. The router stays
frozen. Evaluation uses thresholds 8, 19, and 25 and crosses both
physical layouts, both roles, and saved readings one above or below the
live reading. The four conditions cross a local rank-one comparison
patch with flipping the binding instruction. Donor feedback comes
from the same guest, on the opposite side of the threshold. The host
dispatches with the original pair in all four conditions.

| Condition | Tool correct | Slot correct | Request and result correct | Live value nearest to probe estimate |
| --- | ---: | ---: | ---: | ---: |
| Unmodified | 144/144 | 144/144 | 144/144 | 144/144 |
| Binding instruction only | 144/144 | 144/144 | 144/144 | 144/144 |
| Comparison edit only | 144/144 | 144/144 | 144/144 | 72/144 |
| Both | 144/144 | 144/144 | 144/144 | 72/144 |

The slot score includes abstentions as a diagnostic. A correct abstention
emits no WASM follow-up. The v99 probe is refit on the same calibration
thresholds and reads the first-block feedback token. Its drop shows that
this numeric readout is sensitive to the patch. The trained head can still
select the correct explicit slot, using the role and layout it receives.

The archived example starts with live 7 and saved 6. The baseline calls
Trim(7) and returns 5. Flipping only the binding instruction calls Trim(6)
and returns 4. Editing only the comparison calls Boost(7) and returns 12.
Doing both calls Boost(6) and returns 11. The sweep found no request
mismatch, so the archive opens this successful four-way comparison.
All 18 distinct first calls and 24 distinct follow-ups agree with
Wasmtime on returns and full-page digests, with packed fallback blocked.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_tool_argument_binding.py
python tools/visuaml_replay.py open release/examples/tool-argument-binding/example-11.json
```

[Raw requests, guest source and binary, head weights, and replay traces](../examples/tool-argument-binding/)
