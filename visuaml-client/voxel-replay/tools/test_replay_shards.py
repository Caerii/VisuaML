"""Check ordering and first-difference provenance across worker layouts."""
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from visuaml.replay.shards import merge_shards
from visuaml.replay.pipeline import observation

CLI = Path(__file__).resolve().with_name("visuaml_replay.py")


def rows(*requests):
    return "".join(json.dumps(observation(request, boundary, request, runtime="test"))+"\n"
                   for request in requests for boundary in ("request", "result"))


class ShardTests(unittest.TestCase):
    def test_order_and_diagnostics(self):
        merged = list(merge_shards([StringIO(rows("002", "003")), StringIO(rows("001"))]))
        self.assertEqual([row["request"] for row in merged], ["001"]*2+["002"]*2+["003"]*2)
        self.assertEqual(merged[0]["source"], {"shard": 1, "line": 1})
        with self.assertRaisesRegex(ValueError, "appears in shards"):
            list(merge_shards([StringIO(rows("001")), StringIO(rows("001"))]))
        with self.assertRaisesRegex(ValueError, "Shard 0: Line 3"):
            list(merge_shards([StringIO(rows("002", "001"))]))

    def test_cli_different_partition_and_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / str(index) for index in range(4)]
            for path, body in zip(paths, (rows("001", "003"), rows("002"), rows("002", "003"), rows("001"))):
                path.write_text(body)
            cmd = [sys.executable, str(CLI), "compare-shards", "--reference", *map(str, paths[:2]),
                   "--candidate", *map(str, paths[2:]), "--fail-on-difference"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["matched_observations"], 6)
            paths[-1].write_text(rows("001").replace('"value": "001"', '"value": "wrong"', 1))
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            difference = json.loads(result.stdout)["first_difference"]
            self.assertEqual(difference["candidate"]["source"], {"shard": 1, "line": 1})


if __name__ == "__main__":
    unittest.main()
