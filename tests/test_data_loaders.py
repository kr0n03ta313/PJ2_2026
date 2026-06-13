import sys
import unittest
from pathlib import Path

from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BN_CODE_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(BN_CODE_ROOT))


class ToyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return index, index * 10


class PartialDatasetTests(unittest.TestCase):
    def test_partial_dataset_forwards_index_and_limits_length(self):
        from data.loaders import PartialDataset

        dataset = PartialDataset(ToyDataset(), n_items=3)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[2], (2, 20))


if __name__ == "__main__":
    unittest.main()
