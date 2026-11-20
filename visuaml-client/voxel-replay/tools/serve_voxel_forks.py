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

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
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
