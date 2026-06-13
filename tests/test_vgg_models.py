import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BN_CODE_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(BN_CODE_ROOT))


class VGGModelTests(unittest.TestCase):
    def test_vgg_a_imports_and_outputs_cifar10_logits(self):
        from models.vgg import VGG_A

        model = VGG_A()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (2, 10))

    def test_vgg_a_batchnorm_contains_bn_layers_and_outputs_cifar10_logits(self):
        from models.vgg import VGG_A_BatchNorm

        model = VGG_A_BatchNorm()
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            logits = model(x)

        self.assertGreaterEqual(len(bn_layers), 1)
        self.assertEqual(logits.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
