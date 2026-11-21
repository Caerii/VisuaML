# Emit WASM request bytes from tagged results

The first guest returns four little-endian bytes in the format
`[tag, value, tag, value]`. Tag 0 means live and tag 1 means saved.
Both record orders occur. The guest increments the two value fields
without changing the tags for the tested inputs, stores the packed word
at memory byte 32, and reads it back.

The host passes the two records, threshold, routing rule, and requested
role to the controller. A learned tag scorer attends to both records.
The reader combines the value bits with those attention weights and
thresholds eight output logits to emit the argument byte. It also emits
the selected tag. The same reader supplies the live value to the frozen
seed 829 router. The host dispatches the emitted byte directly. The
controller receives no separate slot layout. The value path explicitly
supports copying input bits, and the benchmark does not test free-form
numeric generation.

The scorer trains on adjacent values near live readings 4–24. The routing
evaluation uses thresholds 8, 19, and 25. Some byte values also occur in
scorer training. The model weights are saved before evaluation. The
comparison patch modifies the router's first-block activations with the
original records held fixed. The binding intervention changes the
requested role. Neither changes the WASM result bytes held by the host.

| Condition | Correct tools | Correct roles | Exact bytes | Correct requests and results |
| --- | ---: | ---: | ---: | ---: |
| Natural | 144/144 | 144/144 | 144/144 | 144/144 |
| Role change | 144/144 | 144/144 | 144/144 | 144/144 |
| Comparison edit | 144/144 | 144/144 | 144/144 | 144/144 |
| Both changes | 144/144 | 144/144 | 144/144 | 144/144 |

The archive includes the first mismatch if one occurs. This run found
none and records a four-way example instead. The 24 distinct first and
24 distinct follow-up WASM calls match Wasmtime on returns and complete
memory-page digests with packed fallback blocked. A correct abstention
issues no follow-up.

From `visuaml-client/voxel-replay` in the research Python environment:

```sh
export PYTHONPATH=.:../backend
python tools/replay_tagged_byte_requests.py
python tools/visuaml_replay.py open release/examples/tagged-byte-requests/example-11.json
```

[Guest, learned scorer weights, raw requests, bit logits, and traces](../examples/tagged-byte-requests/)
