import sys
import contextlib
from functools import lru_cache
import os

import torch
#from modules import errors

if sys.platform == "darwin":
    from modules import mac_specific


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


def get_cuda_device_string():
    return "cuda"


def get_optimal_device_name():
    # 检查环境变量或全局设置来确定设备
    if hasattr(torch, '_hypir_device_override'):
        return torch._hypir_device_override
    
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    return get_optimal_device()


def torch_gc():
    if torch.cuda.is_available() and get_optimal_device_name() != "cpu":
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()


def enable_tf32():
    if torch.cuda.is_available() and get_optimal_device_name() != "cpu":
        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any(torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


enable_tf32()
#errors.run(enable_tf32, "Enabling TF32")

cpu = torch.device("cpu")

# 动态设置设备和数据类型
def _get_device_config():
    device_name = get_optimal_device_name()
    if device_name == "cpu":
        return {
            'device': torch.device("cpu"),
            'dtype': torch.float32,  # CPU模式使用float32获得更好性能
            'dtype_vae': torch.float32,
            'dtype_unet': torch.float32,
            'unet_needs_upcast': False
        }
    else:
        return {
            'device': torch.device(device_name),
            'dtype': torch.float16,
            'dtype_vae': torch.float16,
            'dtype_unet': torch.float16,
            'unet_needs_upcast': False
        }

# 获取配置
_config = _get_device_config()
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = _config['device']
dtype = _config['dtype']
dtype_vae = _config['dtype_vae']
dtype_unet = _config['dtype_unet']
unet_needs_upcast = _config['unet_needs_upcast']


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


def randn(seed, shape):
    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    return torch.randn(shape, device=device)


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or device.type == "cpu":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if not torch.all(torch.isfinite(x)):
        if where == "unet":
            message = "A tensor with all NaNs was produced in Unet."

            if not torch.all(torch.isfinite(x)).all():
                message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

        elif where == "vae":
            message = "A tensor with all NaNs was produced in VAE."

            if not torch.all(torch.isfinite(x)).all():
                message += " This could be because your VAE does not support half type. Try enabling the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."
        else:
            message = "A tensor with all NaNs was produced."

        message += " Use --disable-nan-check commandline argument to disable this check."

        raise NansException(message)


@lru_cache
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocaltes about 700MB of memory and
    spends about 2.7 seconds doing that, at least wih NVidia.
    """

    x = torch.zeros((1, 1)).to(device, dtype)
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    linear(x)

    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    conv2d(x)


def set_device_override(device_name):
    """设置设备覆盖，用于强制使用特定设备"""
    torch._hypir_device_override = device_name
    # 重新配置全局变量
    global device, device_interrogate, device_gfpgan, device_esrgan, device_codeformer
    global dtype, dtype_vae, dtype_unet, unet_needs_upcast
    
    _config = _get_device_config()
    device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = _config['device']
    dtype = _config['dtype']
    dtype_vae = _config['dtype_vae']
    dtype_unet = _config['dtype_unet']
    unet_needs_upcast = _config['unet_needs_upcast']
    
    print(f"设备已设置为: {device}, 数据类型: {dtype}")
