"""Verify that hooks edit the selected output, once, and always clean up."""
import unittest

import torch
from torch import nn

from visuaml.replay.interventions import ActivationSite, run_intervention


class InterventionTests(unittest.TestCase):
    def test_live_output_and_cleanup(self):
        model = nn.Sequential(nn.ReLU(), nn.Identity())
        source = torch.tensor([-1.0, 2.0])
        edit = torch.tensor([3.0, 4.0])
        result, before, after = run_intervention(model, ActivationSite("0"), (source,), replacement=edit)
        self.assertEqual(result.tolist(), [3, 4])
        self.assertEqual(before.tolist(), [0, 2])
        self.assertEqual(after.tolist(), [3, 4])
        with torch.inference_mode():
            after.zero_()
        self.assertEqual(edit.tolist(), [3, 4])
        with self.assertRaisesRegex(ValueError, "match shape"):
            run_intervention(model, ActivationSite("0"), (source,), replacement=edit[:1])
        self.assertEqual(model(source).tolist(), [0, 2])

    def test_repeated_module_is_rejected(self):
        class Reused(nn.Module):
            def __init__(self):
                super().__init__()
                self.same = nn.Identity()

            def forward(self, value):
                return self.same(self.same(value))

        model = Reused()
        with self.assertRaisesRegex(ValueError, "more than once"):
            run_intervention(model, ActivationSite("same"), (torch.zeros(1),))


if __name__ == "__main__":
    unittest.main()
