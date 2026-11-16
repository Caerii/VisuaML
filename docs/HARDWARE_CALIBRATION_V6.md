# Latency calibration contract v6 (experimental)

`visuaml.compiler_lab.hardware_calibration` accepts typed latency observations
for **one** declared device, operator and evidence-source category at a time.
It fits a simple affine cost model with explicitly named units:

`predicted_latency_us = baseline_us + us_per_work_unit * work_units + us_per_byte * bytes_moved`

The input dataset is sorted by measurement ID before hashing, so the SHA-256
fingerprint is independent of input order but sensitive to provenance and
measurements. Calibration records keep a `source_kind` of `synthetic` or
`measured`, and the API refuses to mix them. The library does NOT verify
whether an observation claimed as measured was actually made on hardware.
This must eventually be backed by hardware-owned signed measurement receipts.

The fit requires 6 or more distinct observations and a full-rank design matrix.
A negative fitted physical cost is refused. The report exposes mean and maximum
absolute residual error. Predictions outside observed workload/transfer ranges
are rejected unless extrapolation is explicitly enabled.

No Ayo measurements, photonic results, or physical energy/latency claims are
included. The accompanying tests use clearly labeled synthetic data to verify
input admission, parameter recovery, provenance hashing, diagnostics and
extrapolation barriers.

Next steps: ingest actual benchmark capture files with operator signatures and
complete pipeline costs (conversion, lasers, electronics, I/O), evaluate a
held-out validation set, estimate uncertainty, then feed calibrated costs into
the placement optimizer. Without those measurements, current hardware costs
remain strictly hypothetical.

Run from `visuaml-client`:

```sh
python -m pytest -q backend/tests/test_hardware_calibration_v6.py
```
