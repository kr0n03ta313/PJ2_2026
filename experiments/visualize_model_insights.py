import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from run_project2_experiments import CLASSES, build_model


def collect_feature_maps(model, x, layer_names):
    activations = {}
    handles = []
    named_modules = dict(model.named_modules())
    for name in layer_names:
        if name not in named_modules:
            raise ValueError(f"Layer not found: {name}")

        def hook(_, __, output, layer_name=name):
            activations[layer_name] = output.detach().cpu()

        handles.append(named_modules[name].register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for handle in handles:
        handle.remove()
    return activations


def build_confusion_matrix(y_true, y_pred, num_classes):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, pred in zip(y_true.view(-1), y_pred.view(-1)):
        matrix[int(target), int(pred)] += 1
    return matrix


def denormalize(x):
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def make_test_loader(data_root, batch_size=128, max_items=1000):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
    if max_items and max_items < len(dataset):
        dataset = Subset(dataset, list(range(max_items)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def plot_feature_maps(features, output_path, max_channels=16):
    rows = len(features)
    fig, axes = plt.subplots(rows, max_channels, figsize=(max_channels * 1.1, rows * 1.3))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, (name, fmap) in enumerate(features.items()):
        channels = fmap[0, :max_channels]
        for col_idx in range(max_channels):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < channels.size(0):
                ax.imshow(channels[col_idx], cmap="viridis")
            if col_idx == 0:
                ax.set_title(name, fontsize=8)
    fig.suptitle("Intermediate feature maps")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix, output_path):
    normalized = matrix.float() / matrix.sum(dim=1, keepdim=True).clamp_min(1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(normalized.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = normalized[i, j].item()
            if value > 0.05:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prediction_examples(images, labels, preds, output_path, max_items=16):
    count = min(max_items, images.size(0))
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if idx >= count:
            continue
        img = denormalize(images[idx]).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        color = "green" if int(labels[idx]) == int(preds[idx]) else "red"
        ax.set_title(f"T:{CLASSES[int(labels[idx])]}\nP:{CLASSES[int(preds[idx])]}", color=color, fontsize=8)
    fig.suptitle("Prediction examples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_visualizations(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = PROJECT_ROOT / args.output_dir
    model = build_model(args.model).to(device)
    state = torch.load(PROJECT_ROOT / args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = make_test_loader(PROJECT_ROOT / args.data_root, batch_size=args.batch_size, max_items=args.max_items)
    all_targets = []
    all_preds = []
    first_images = None
    first_labels = None
    first_preds = None
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
        if first_images is None:
            first_images = images.detach().cpu()
            first_labels = labels.detach().cpu()
            first_preds = preds.detach().cpu()
        all_targets.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())

    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_preds)
    matrix = build_confusion_matrix(y_true, y_pred, len(CLASSES))
    plot_confusion_matrix(matrix, output_dir / "final_model_confusion_matrix.png")
    plot_prediction_examples(first_images, first_labels, first_preds, output_dir / "final_model_prediction_examples.png")

    sample = first_images[:1].to(device)
    features = collect_feature_maps(model, sample, args.layers)
    plot_feature_maps(features, output_dir / "final_model_feature_maps.png")

    accuracy = (y_true == y_pred).float().mean().item()
    (output_dir / "final_model_insight_summary.txt").write_text(
        f"items={len(y_true)}\naccuracy={accuracy:.4f}\nlayers={','.join(args.layers)}\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/cifar10")
    parser.add_argument("--weights", default="reports/project2_experiments/runs/final_small_relu_sgd_full_5ep/model.pt")
    parser.add_argument("--output-dir", default="reports/project2_experiments/figures")
    parser.add_argument("--model", default="small_relu")
    parser.add_argument("--layers", nargs="+", default=["features.0", "features.4"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    generate_visualizations(parse_args())
