import json
from pathlib import Path

import torch


def count_parameters(model, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


def save_checkpoint(model, path, extra_info=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "extra_info": extra_info or {},
        },
        path,
    )


def load_checkpoint(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_model_info(path, model, extra_info=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    info = {
        "trainable_parameters": count_parameters(model, True),
        "total_parameters": count_parameters(model, False),
        "model_size_mb": round(model_size_mb(model), 4),
    }

    if extra_info:
        info.update(extra_info)

    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
