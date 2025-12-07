"""Lightweight MNIST training harness with projective augmentations."""
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 2
    lr: float = 3e-3
    train_size: int | None = 5000
    test_size: int | None = 1000
    use_augmentation: bool = True
    num_workers: int = 2


def make_transforms(augment: bool) -> Callable:
    aug_transforms: list[transforms.Transform] = []
    if augment:
        aug_transforms.extend(
            [
                transforms.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.35, p=1.0),
            ]
        )
    aug_transforms.append(transforms.ToTensor())
    return transforms.Compose(aug_transforms)


def make_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=make_transforms(cfg.use_augmentation)
    )
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    if cfg.train_size:
        train_data = Subset(train_data, range(cfg.train_size))
    if cfg.test_size:
        test_data = Subset(test_data, range(cfg.test_size))

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, test_loader


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        logits = out["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        logits = out["logits"]
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(cfg)

    model = build_model(image_size=(28, 28), num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Device: {device}")
    print(f"Augmentation: {cfg.use_augmentation}")
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%"
        )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-augment", action="store_true", help="Disable projective and affine augmentation")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    return TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        train_size=args.train_size,
        test_size=args.test_size,
        use_augmentation=not args.no_augment,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    train(parse_args())
