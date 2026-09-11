from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


EMNIST_MEAN = 0.1307
EMNIST_STD = 0.3081


# ---------------------------------------------------------
# Training transformation
# ---------------------------------------------------------
# Augmentation is applied ONLY to training images.
# This helps the model learn different handwriting variations.
# ---------------------------------------------------------

EMNIST_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.10, 0.10),
        scale=(0.90, 1.10),
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (EMNIST_MEAN,),
        (EMNIST_STD,),
    ),
])


# ---------------------------------------------------------
# Validation / Test transformation
# ---------------------------------------------------------
# No random augmentation here.
# Validation and test data must remain unchanged.
# ---------------------------------------------------------

EMNIST_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (EMNIST_MEAN,),
        (EMNIST_STD,),
    ),
])


def get_emnist_datasets(
    data_dir="data/raw",
    validation_size=10000,
    seed=42,
):
    data_dir = Path(data_dir)

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Full training dataset
    # -----------------------------------------------------

    full_train_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=True,
        download=True,
        transform=EMNIST_TRAIN_TRANSFORM,
    )

    # -----------------------------------------------------
    # Separate evaluation dataset
    # -----------------------------------------------------

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=False,
        download=True,
        transform=EMNIST_EVAL_TRANSFORM,
    )

    # -----------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------

    train_size = (
        len(full_train_dataset)
        - validation_size
    )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [
            train_size,
            validation_size,
        ],
        generator=generator,
    )

    # -----------------------------------------------------
    # IMPORTANT
    # -----------------------------------------------------
    # random_split uses the same underlying dataset.
    #
    # Therefore validation would also receive the training
    # augmentation if we simply returned it.
    #
    # We create a second EMNIST dataset with evaluation
    # transforms and reuse the validation indices.
    # -----------------------------------------------------

    full_eval_dataset = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=True,
        download=False,
        transform=EMNIST_EVAL_TRANSFORM,
    )

    validation_dataset = torch.utils.data.Subset(
        full_eval_dataset,
        validation_dataset.indices,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def get_emnist_dataloaders(
    data_dir="data/raw",
    batch_size=128,
    validation_size=10000,
    num_workers=0,
    seed=42,
):
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

    return (
        train_loader,
        validation_loader,
        test_loader,
    )