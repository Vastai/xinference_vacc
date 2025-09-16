from typing import Literal
import torch
import os

DeviceType = Literal["vacc","cuda", "mps", "xpu", "npu", "cpu"]
DEVICE_TO_ENV_NAME = {
    "vacc": "VACC_VISIBLE_DEVICES",
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
}

def is_vacc_available() -> bool:
    try:
        return torch.vacc.is_available()
    except ImportError:
        return False

def get_available_device() -> DeviceType:
    # NOTE(lancew): must first check if vacc is available
    if is_vacc_available():
        return "vacc"
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"

def gpu_count():
    # NOTE(lancew): must first check if vacc is available
    if is_vacc_available():
        vacc_visible_devices_env = os.getenv("VACC_VISIBLE_DEVICES", None)
        if vacc_visible_devices_env is None:
            return torch.vacc.device_count()
        vacc_visible_devices = (
            vacc_visible_devices_env.split(",") if vacc_visible_devices_env else []
        )
        return min(torch.vacc.device_count(), len(vacc_visible_devices))
    elif torch.cuda.is_available():
        cuda_visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible_devices_env is None:
            return torch.cuda.device_count()

        cuda_visible_devices = (
            cuda_visible_devices_env.split(",") if cuda_visible_devices_env else []
        )

        return min(torch.cuda.device_count(), len(cuda_visible_devices))
    elif is_xpu_available():
        return torch.xpu.device_count()
    elif is_npu_available():
        return torch.npu.device_count()
    else:
        return 0