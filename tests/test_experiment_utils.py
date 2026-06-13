import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))


class ExperimentUtilsTests(unittest.TestCase):
    def test_compute_loss_band_aligns_runs_by_shortest_length(self):
        from run_project2_experiments import compute_loss_band

        min_curve, max_curve = compute_loss_band([[3.0, 2.0, 1.0], [4.0, 1.5]])

        self.assertEqual(min_curve, [3.0, 1.5])
        self.assertEqual(max_curve, [4.0, 2.0])

    def test_count_parameters_returns_trainable_parameter_count(self):
        from run_project2_experiments import count_parameters

        model = torch.nn.Linear(3, 2)

        self.assertEqual(count_parameters(model), 8)

    def test_model_factory_creates_required_models(self):
        from run_project2_experiments import build_model

        for name in ["small_relu", "small_leaky_relu", "small_elu", "vgg_a", "vgg_a_bn", "vgg_dropout"]:
            model = build_model(name)
            with torch.no_grad():
                logits = model(torch.randn(2, 3, 32, 32))
            self.assertEqual(logits.shape, (2, 10))

    def test_write_json_creates_parent_directory(self):
        from run_project2_experiments import write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "metrics.json"
            write_json(path, {"accuracy": 0.5})

            self.assertTrue(path.exists())
            self.assertIn("accuracy", path.read_text(encoding="utf-8"))

    def test_run_config_keeps_positional_epoch_arguments_stable(self):
        from run_project2_experiments import RunConfig

        config = RunConfig("bn", "vgg_a", "adam", 0.001, 0.0, 3, 128, 4000, 10000)

        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.label_smoothing, 0.0)


if __name__ == "__main__":
    unittest.main()
