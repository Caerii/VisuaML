"""Loopback-only demo endpoint for the fixed shipped fork search."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from unittest.mock import patch

from replay_frozen_recurrent import PACKED
from search_voxel_forks import Witness, make_checkpoint, search


def main():
    witness = Witness()
    with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
        checkpoint = make_checkpoint(witness)

    trace_relay = None
    world_inference = None

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal trace_relay, world_inference
            if self.path in ("/api/pocket-world-reconstruct/near","/api/pocket-world-reconstruct/delayed"):
                import torch
                from replay_pocket_world_inference import OUT, reconstruct
                with torch.inference_mode():
                    if world_inference is None:
                        torch.set_num_threads(1)
                        world_inference = torch.jit.load(str(OUT / "inference.pt"))
                    self.reply(reconstruct(self.path.rsplit("/",1)[-1],world_inference))
                return
            if self.path == "/api/pocket-reconstruct":
                import torch
                from replay_on_demand_trace import DEMAND, OUT, Relay, reconstruct
                packet = json.loads((OUT / "pending.json").read_text())
                with torch.inference_mode(), patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
                    if trace_relay is None:
                        trace_relay = Relay(DEMAND / "signal-relay.wasm")
                    self.reply(reconstruct(packet, trace_relay))
                return
            if self.path == "/api/compiler-counterexample":
                from search_compiler_counterexample import Experiment
                with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
                    experiment = Experiment("terrain")
                    result = experiment.search(experiment.initial(), True)
                self.reply(result)
                return
            if self.path != "/api/voxel-fork-search":
                self.send_error(404)
                return
            with patch(PACKED, side_effect=AssertionError("Unexpected packed fallback")):
                result = search(witness, checkpoint)
            self.reply({key: result[key] for key in ("winner", "metrics")})

        def reply(self, result):
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with HTTPServer(("127.0.0.1", 4181), Handler) as server:
        print("VisuaML fork search listening on http://127.0.0.1:4181", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
