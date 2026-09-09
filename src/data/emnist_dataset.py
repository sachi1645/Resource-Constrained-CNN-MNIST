from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


EMNIST_MEAN = 0.1307
EMNIST_STD = 0.3081


EMNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (EMNIST_MEAN,),
        (EMNIST_STD,)
    ),
])


def get_emnist_datasets(
    data_dir="data/raw",
    validation_size=10000,
    seed=42,
):
    """
    Download and prepare the EMNIST Balanced dataset.

    EMNIST Balanced contains 47 balanced character classes.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    full_train_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=True,
        download=True,
        transform=EMNIST_TRANSFORM,
    )

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=False,
        download=True,
        transform=EMNIST_TRANSFORM,
    )

    train_size = len(full_train_dataset) - validation_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=generator,
    )

    return train_dataset, validation_dataset, test_dataset


def get_emnist_dataloaders(
    data_dir="data/raw",
    batch_size=128,
    validation_size=10000,
    num_workers=0,
    seed=42,
):
    """
    Create DataLoaders for EMNIST Balanced.
    """

    train_dataset, validation_dataset, test_dataset = (
        get_emnist_datasets(
            data_dir=data_dir,
            validation_size=validation_size,
            seed=seed,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader, test_loader