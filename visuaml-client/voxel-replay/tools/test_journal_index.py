"""Compare indexed results against the ordered scanner, including corruption."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from visuaml.replay.journal_index import BLOCK_ROWS, build_index, compare_indexes, verify_index
from visuaml.replay.shards import compare_shards


def write(path, values, runtime):
    with path.open("w") as stream:
        for i, value in enumerate(values):
            stream.write(json.dumps(dict(request=f"{i:06}", boundary="result", runtime=runtime, value=value))+"\n")


class JournalIndexTests(unittest.TestCase):
    def test_exact_parity(self):
        values = [0.0]*(BLOCK_ROWS*2+5)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            left, right, a, b = (root/name for name in ("left", "right", "a", "b"))
            write(left, values, "A")
            build_index([left], a)
            for candidate in (values, [*values[:-1], 9], values[:-1], [*values, 1]):
                write(right, candidate, "B")
                build_index([right], b)
                original = compare_shards([left], [right])
                fast = compare_indexes(a, b)
                self.assertEqual(fast["matched_observations"], original["matched_observations"])
                self.assertEqual(fast["first_difference"], original["first_difference"])
                self.assertGreaterEqual(fast["skipped_blocks"], 2)
            write(right, [.1, *values[1:-1], 9], "B")
            build_index([right], b)
            self.assertEqual(compare_indexes(a, b, tolerances={"result": (0, .2)})[
                "first_difference"]["index"], len(values)-1)

    def test_corruption_and_incomplete_publication(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            left, right, a, b = (root/name for name in ("left", "right", "a", "b"))
            write(left, [0], "A")
            write(right, [1], "B")
            build_index([left], a)
            build_index([right], b)
            self.assertEqual(verify_index(b)["observations"], 1)
            original = b.read_bytes()
            right.write_text("bad input\n")
            with self.assertRaises(ValueError):
                build_index([right], b)
            self.assertEqual(b.read_bytes(), original)
            changed = bytearray(original)
            changed[0] ^= 1
            b.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "block integrity"):
                compare_indexes(a, b)
            with self.assertRaisesRegex(ValueError, "block integrity"):
                verify_index(b)
            changed = bytearray(original)
            changed[-57] ^= 1
            b.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "metadata integrity"):
                compare_indexes(a, b)


if __name__ == "__main__":
    unittest.main()
