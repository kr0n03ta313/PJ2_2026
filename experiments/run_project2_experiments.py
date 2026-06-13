import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BN_CODE_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(BN_CODE_ROOT))

from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class RunConfig:
    name: str
    model: str
    optimizer: str
    lr: float
    weight_decay: float = 0.0
    epochs: int = 1
    batch_size: int = 256
    train_items: int = 12000
    test_items: int = 10000
    seed: int = 2026
    label_smoothing: float = 0.0


class SmallCifarCNN(nn.Module):
    def __init__(self, activation="relu", filters=(32, 64), dropout=0.0):
        super().__init__()
        act = make_activation(activation)
        self.features = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            act(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            act(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters[1] * 8 * 8, 256),
            act(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_activation(name):
    if name == "relu":
        return lambda: nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "elu":
        return lambda: nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def build_model(name):
    if name == "small_relu":
        return SmallCifarCNN("relu", filters=(32, 64), dropout=0.1)
    if name == "small_leaky_relu":
        return SmallCifarCNN("leaky_relu", filters=(32, 64), dropout=0.1)
    if name == "small_elu":
        return SmallCifarCNN("elu", filters=(32, 64), dropout=0.1)
    if name == "small_wide":
        return SmallCifarCNN("relu", filters=(64, 128), dropout=0.1)
    if name == "small_dropout":
        return SmallCifarCNN("relu", filters=(32, 64), dropout=0.4)
    if name == "vgg_light":
        return VGG_A_Light()
    if name == "vgg_a":
        return VGG_A()
    if name == "vgg_a_bn":
        return VGG_A_BatchNorm()
    if name == "vgg_dropout":
        return VGG_A_Dropout()
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_loss_band(loss_runs):
    usable = [run for run in loss_runs if run]
    if not usable:
        return [], []
    length = min(len(run) for run in usable)
    stacked = np.array([run[:length] for run in usable], dtype=np.float64)
    return stacked.min(axis=0).tolist(), stacked.max(axis=0).tolist()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loaders(data_root, batch_size, train_items, test_items, workers):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
    if 0 < train_items < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(train_items)))
    if 0 < test_items < len(test_dataset):
        test_dataset = Subset(test_dataset, list(range(test_items)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def make_optimizer(name, model, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_total += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    accuracy = correct / total
    return {"loss": loss_total / total, "accuracy": accuracy, "error": 1.0 - accuracy}


def train_one(config, data_root, output_root, workers, device):
    set_seed(config.seed)
    train_loader, test_loader = make_loaders(
        data_root=data_root,
        batch_size=config.batch_size,
        train_items=config.train_items,
        test_items=config.test_items,
        workers=workers,
    )
    model = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = make_optimizer(config.optimizer, model, config.lr, config.weight_decay)

    step_losses = []
    epoch_metrics = []
    start = time.time()
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            step_losses.append(float(loss.item()))
            epoch_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        test_metrics = evaluate(model, test_loader, device)
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / total,
                "train_accuracy": correct / total,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_error": test_metrics["error"],
            }
        )
    elapsed = time.time() - start
    output_dir = output_root / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)
    result = {
        "config": asdict(config),
        "parameters": count_parameters(model),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "step_losses": step_losses,
        "epochs": epoch_metrics,
        "best_test_accuracy": max(item["test_accuracy"] for item in epoch_metrics),
        "best_test_error": min(item["test_error"] for item in epoch_metrics),
        "weights_path": str(weights_path.relative_to(PROJECT_ROOT)),
    }
    write_json(output_dir / "metrics.json", result)
    return result


def plot_training_curves(results, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for result in results:
        epochs = [item["epoch"] for item in result["epochs"]]
        accuracies = [item["test_accuracy"] for item in result["epochs"]]
        plt.plot(epochs, accuracies, marker="o", label=result["config"]["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR-10 test accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = output_root / "test_accuracy_curves.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_loss_landscape(name, grouped_runs, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    bands = {}
    for label, runs in grouped_runs.items():
        min_curve, max_curve = compute_loss_band([run["step_losses"] for run in runs])
        bands[label] = {"min_curve": min_curve, "max_curve": max_curve}
        steps = list(range(len(min_curve)))
        plt.plot(steps, min_curve, linewidth=1.0, label=f"{label} min")
        plt.plot(steps, max_curve, linewidth=1.0, label=f"{label} max")
        plt.fill_between(steps, min_curve, max_curve, alpha=0.2)
    plt.xlabel("Training step")
    plt.ylabel("Batch loss")
    plt.title("Loss landscape band from learning-rate sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    figure_path = output_root / f"{name}.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()
    write_json(output_root / f"{name}_bands.json", bands)
    return figure_path


def default_configs(epochs, train_items, test_items, batch_size):
    common = {"epochs": epochs, "train_items": train_items, "test_items": test_items, "batch_size": batch_size}
    return [
        RunConfig("small_relu_adam", "small_relu", "adam", 1e-3, 0.0, **common),
        RunConfig("small_wide_adam", "small_wide", "adam", 1e-3, 0.0, **common),
        RunConfig("small_dropout_adam", "small_dropout", "adam", 1e-3, 0.0, **common),
        RunConfig("small_relu_adamw_wd", "small_relu", "adamw", 1e-3, 1e-4, **common),
        RunConfig("small_relu_sgd", "small_relu", "sgd", 5e-2, 1e-4, **common),
        RunConfig("small_leaky_relu_adam", "small_leaky_relu", "adam", 1e-3, 0.0, **common),
        RunConfig("small_elu_adam", "small_elu", "adam", 1e-3, 0.0, **common),
        RunConfig("vgg_light_adam", "vgg_light", "adam", 1e-3, 0.0, **common),
    ]


def bn_configs(epochs, train_items, test_items, batch_size):
    rates = [1e-3, 2e-3, 1e-4, 5e-4]
    configs = []
    for lr in rates:
        configs.append(
            RunConfig(
                f"bn_sweep_vgg_a_lr_{lr:g}",
                "vgg_a",
                "adam",
                lr,
                0.0,
                epochs,
                batch_size,
                train_items,
                test_items,
            )
        )
        configs.append(
            RunConfig(
                f"bn_sweep_vgg_a_bn_lr_{lr:g}",
                "vgg_a_bn",
                "adam",
                lr,
                0.0,
                epochs,
                batch_size,
                train_items,
                test_items,
            )
        )
    return configs


def run_all(args):
    output_root = PROJECT_ROOT / args.output
    results_dir = output_root / "runs"
    figures_dir = output_root / "figures"
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    configs = default_configs(args.epochs, args.train_items, args.test_items, args.batch_size)
    bn_sweep = bn_configs(args.bn_epochs, args.bn_train_items, args.test_items, args.batch_size)
    all_results = []
    for config in configs + bn_sweep:
        print(f"[run] {config.name}", flush=True)
        all_results.append(train_one(config, PROJECT_ROOT / args.data_root, results_dir, args.workers, device))

    main_results = all_results[: len(configs)]
    bn_results = all_results[len(configs) :]
    plot_training_curves(main_results, figures_dir)
    grouped = {
        "VGG-A": [item for item in bn_results if item["config"]["model"] == "vgg_a"],
        "VGG-A+BN": [item for item in bn_results if item["config"]["model"] == "vgg_a_bn"],
    }
    plot_loss_landscape("bn_loss_landscape", grouped, figures_dir)
    summary = {
        "device": str(device),
        "main_results": [
            {
                "name": item["config"]["name"],
                "model": item["config"]["model"],
                "optimizer": item["config"]["optimizer"],
                "lr": item["config"]["lr"],
                "weight_decay": item["config"]["weight_decay"],
                "parameters": item["parameters"],
                "best_test_accuracy": item["best_test_accuracy"],
                "best_test_error": item["best_test_error"],
                "elapsed_seconds": item["elapsed_seconds"],
                "weights_path": item["weights_path"],
            }
            for item in main_results
        ],
        "bn_results": [
            {
                "name": item["config"]["name"],
                "model": item["config"]["model"],
                "lr": item["config"]["lr"],
                "parameters": item["parameters"],
                "best_test_accuracy": item["best_test_accuracy"],
                "best_test_error": item["best_test_error"],
                "elapsed_seconds": item["elapsed_seconds"],
                "weights_path": item["weights_path"],
            }
            for item in bn_results
        ],
        "figures": {
            "test_accuracy_curves": str((figures_dir / "test_accuracy_curves.png").relative_to(PROJECT_ROOT)),
            "bn_loss_landscape": str((figures_dir / "bn_loss_landscape.png").relative_to(PROJECT_ROOT)),
        },
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/cifar10")
    parser.add_argument("--output", default="reports/project2_experiments")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bn-epochs", type=int, default=1)
    parser.add_argument("--train-items", type=int, default=12000)
    parser.add_argument("--bn-train-items", type=int, default=6000)
    parser.add_argument("--test-items", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
