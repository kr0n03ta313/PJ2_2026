import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))


class VisualizationUtilsTests(unittest.TestCase):
    def test_collect_feature_maps_returns_named_layers(self):
        from visualize_model_insights import collect_feature_maps

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 6, 3, padding=1),
            torch.nn.ReLU(),
        )
        x = torch.randn(1, 3, 8, 8)

        features = collect_feature_maps(model, x, ["0", "2"])

        self.assertEqual(set(features), {"0", "2"})
        self.assertEqual(tuple(features["0"].shape), (1, 4, 8, 8))
        self.assertEqual(tuple(features["2"].shape), (1, 6, 8, 8))

    def test_confusion_matrix_counts_predictions(self):
        from visualize_model_insights import build_confusion_matrix

        y_true = torch.tensor([0, 1, 1, 2])
        y_pred = torch.tensor([0, 1, 2, 2])

        matrix = build_confusion_matrix(y_true, y_pred, num_classes=3)

        self.assertEqual(matrix.tolist(), [[1, 0, 0], [0, 1, 1], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
