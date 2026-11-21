"""Check direct capture, layout independence, and range comparison parity."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from visuaml.replay.pipeline import compare_observations, observation
from visuaml.replay.request_tree import JournalTree, JournalWriter, compare_trees, link_trees, open_tree


def write(path, values, width=4):
    records = []
    with JournalWriter(path, prefix_bytes=width) as writer:
        for line, (key, value) in enumerate(values, 1):
            writer.write(key, "result", value, runtime=path.name)
            records.append(dict(observation(key, "result", value, runtime=path.name),
                                source={"shard": 0, "line": line}))
    return records


class RequestTreeTests(unittest.TestCase):
    def test_parity_and_tolerance(self):
        with TemporaryDirectory() as directory:
            a, b = (Path(directory)/name for name in ("a", "b"))
            values = [(f"{i:06}", 1) for i in range(401)]
            left = write(a, values)
            for changed in (values, values[100:], values[:-1], [*values[:-1], ("000400", 2)]):
                right = write(b, changed)
                with JournalTree(a) as reference, JournalTree(b) as candidate:
                    result = compare_trees(reference, candidate)
                expected = compare_observations(left, right)
                self.assertEqual({k: result[k] for k in expected}, expected)
                if changed is values:
                    self.assertEqual((result["visited_nodes"], result["inspected_leaves"]), (1, 0))
            write(b, [(key, 1.1) for key, _ in values])
            with JournalTree(a) as reference, JournalTree(b) as candidate:
                self.assertIsNone(compare_trees(reference, candidate, tolerances={"result": (0, .2)})[
                    "first_difference"])

    def test_linked_workers_and_replacement(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, flat, manifest = (root/name for name in ("a", "b", "flat", "run.json"))
            left, right = [("000100", 1)], [("000201", 3)]
            write(a, left)
            write(b, right)
            write(flat, left+right)
            with self.assertRaisesRegex(ValueError, "share request prefix"):
                link_trees([a, b], manifest, partition_bytes=3)
            link_trees([b, a], manifest, partition_bytes=4)
            with open_tree(flat) as reference, open_tree(manifest) as candidate:
                result = compare_trees(reference, candidate)
                self.assertEqual(result["matched_observations"], 2)
                self.assertEqual(result["visited_nodes"], 1)
                self.assertIsNone(compare_observations(reference.all_rows(), candidate.all_rows())["first_difference"])
            write(flat, left+[("000201", 9)])
            with open_tree(flat) as reference, open_tree(manifest) as candidate:
                result = compare_trees(reference, candidate)
                self.assertEqual(result["first_difference"]["candidate"]["source"], {"shard": 0, "line": 1})
            write(b, right)
            with self.assertRaisesRegex(ValueError, "root changed"), open_tree(manifest):
                self.fail("Replacement must be rejected even if logical rows match")

    def test_interruption_and_empty(self):
        with TemporaryDirectory() as directory:
            a = Path(directory)/"a"
            with self.assertRaises(RuntimeError), JournalWriter(a) as writer:
                writer.write("a", "result", 1, runtime="x")
                raise RuntimeError("interrupted")
            self.assertFalse(a.exists())
            self.assertFalse(list(a.parent.iterdir()))
            write(a, [])
            with self.assertRaisesRegex(ValueError, "out of order"), JournalWriter(a) as writer:
                writer.write("b", "result", 1, runtime="x")
                writer.write("a", "result", 1, runtime="x")
            with JournalTree(a) as first, JournalTree(a) as second:
                self.assertEqual(compare_trees(first, second)["matched_observations"], 0)


if __name__ == "__main__":
    unittest.main()
