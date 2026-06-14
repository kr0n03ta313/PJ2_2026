# Project 2: CIFAR-10 and Batch Normalization

This repository contains the code for Project 2 of Neural Network and Deep Learning. The experiments train CNN models on CIFAR-10 and compare VGG-A with and without Batch Normalization.

## Code Structure

```text
codes/VGG_BatchNorm/
  data/loaders.py              CIFAR-10 dataloader helper
  models/vgg.py                VGG-A, VGG-A+BN, VGG-A-Dropout, VGG-A-Light
  utils/nn.py                  weight initialization helpers
  VGG_Loss_Landscape.py        original assignment scaffold

experiments/
  run_project2_experiments.py  training, evaluation, BN sweep, plots
  visualize_model_insights.py  filters, feature maps, confusion matrix, examples

tests/
  test_data_loaders.py
  test_experiment_utils.py
  test_vgg_models.py
  test_visualization_utils.py

reports/project2_experiments/
  summary.json                 initial experiment summary
  main5_summary.json           5-epoch model comparison summary
  figures/                     generated result figures
```

## Main Results

- Final model: small CNN with ReLU, BatchNorm, Dropout, SGD momentum, weight decay.
- Training data: full CIFAR-10 training set, 50,000 images.
- Test data: full CIFAR-10 test set, 10,000 images.
- Best test accuracy: 70.61%.
- Best test error: 29.39%.
- BN loss landscape sweep shows a narrower loss band for VGG-A+BN than VGG-A.

## Reproduce

Use the Python environment with PyTorch and torchvision installed.

```powershell
python experiments\run_project2_experiments.py --epochs 5 --bn-epochs 1 --train-items 12000 --bn-train-items 4000 --test-items 10000
python experiments\visualize_model_insights.py --max-items 1000 --layers features.0 features.4
python -m unittest discover -s tests -v
```

The CIFAR-10 dataset and trained model weights are not committed because they are large. The final checkpoint is uploaded to Hugging Face: https://huggingface.co/chris-yu/PJ2_2026/resolve/main/model.pt
