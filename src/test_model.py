import torch


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def gpu_info() -> str:
    device = get_device()

    if device.type == "cuda":
        return torch.cuda.get_device_name(0)

    return "CPU"