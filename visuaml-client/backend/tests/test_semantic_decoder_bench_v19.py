"""Match semantic decoder CPU timing only after full generated-output parity."""

from examples.semantic_decoder_bench_v19 import bench


def test_matched_semantic_cpu_harness_smoke():
    report = bench(layers=1, prompt=4, steps=2, trials=3, seed=19)
    assert report["parity"] == "generated token IDs and every layer final KV pair"
    assert report["construction_ms"] > 0
    assert report["v16_handwritten"]["median_ms"] > 0
    assert report["v18_semantic_cpu"]["median_ms"] > 0
    assert report["logical_memory"]["peak_logical_bytes"] > 0
