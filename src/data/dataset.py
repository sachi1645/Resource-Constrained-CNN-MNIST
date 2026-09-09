from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.data.preprocessing import MNIST_TRANSFORM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_datasets(seed: int = 42) -> Tuple[datasets.MNIST, datasets.MNIST, datasets.MNIST]:
    """Download MNIST and return train, validation and test datasets."""
    full_train = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=MNIST_TRANSFORM,
    )

    train_size = 55_000
    val_size = len(full_train) - train_size

    generator = __import__("torch").Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=MNIST_TRANSFORM,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create train, validation and test DataLoaders."""
    train_dataset, val_dataset, test_dataset = get_datasets(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
