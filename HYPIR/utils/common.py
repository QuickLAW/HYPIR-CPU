import logging
from typing import Mapping, Any, Tuple, Callable, Literal
import importlib
import os
from urllib.parse import urlparse
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
from torch import Tensor
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

from torch.hub import download_url_to_file, get_dir


def get_obj_from_str(string: str, reload: bool=False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/download_util.py/
def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    
    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)]
    weights = np.outer(y_probs, x_probs)
    return weights


@dataclass(frozen=True)
class TileIndex:
    hi: int
    hi_end: int
    wi: int
    wi_end: int


def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    progress: bool = True,
    desc: str=None,
    num_workers: int = 4,  # 新增参数：工作线程数
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Only split the first input of function.
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 多线程处理单个tile的函数
        def process_single_tile(tile_info):
            hi, hi_end, wi, wi_end = tile_info
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(
                scale_fn, (hi, hi_end, wi, wi_end)
            )
            
            # 复制kwargs避免多线程冲突
            tile_kwargs = kwargs.copy()
            if len(args) or len(kwargs):
                tile_kwargs.update(index=TileIndex(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            
            result = fn(x_tile, *args, **tile_kwargs) * weights
            return result, out_hi, out_hi_end, out_wi, out_wi_end
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        # 预分配内存缓冲区，使用连续内存布局提高性能
        out = torch.empty(
            (b, out_channel, scale_fn(h), scale_fn(w)),
            dtype=out_dtype,
            device=out_device,
            memory_format=torch.contiguous_format,  # 使用连续内存格式
        ).zero_()  # 使用zero_()避免重新分配
        
        count = torch.empty_like(out, dtype=torch.float32, memory_format=torch.contiguous_format).zero_()
        # 权重缓存优化，避免重复创建相同的权重
        weight_size = scale_fn(size)
        if weight == "gaussian":
            # 使用缓存的权重或创建新的
            cache_key = f"gaussian_{weight_size}_{str(out_device)}"
            if not hasattr(make_tiled_fn, '_weight_cache'):
                make_tiled_fn._weight_cache = {}
            
            if cache_key not in make_tiled_fn._weight_cache:
                make_tiled_fn._weight_cache[cache_key] = torch.tensor(
                    gaussian_weights(weight_size, weight_size)[None, None],
                    dtype=out_dtype,
                    device=out_device,
                )
            weights = make_tiled_fn._weight_cache[cache_key]
        else:
            # 均匀权重更简单，直接创建
            weights = torch.ones(
                (1, 1, weight_size, weight_size),
                dtype=out_dtype,
                device=out_device,
            )

        indices = sliding_windows(h, w, size, stride)
        
        # 根据设备类型和tile数量选择处理策略
        use_multithread = len(indices) > 4 and (device is None or str(device) == "cpu")
        
        if use_multithread and num_workers > 1:
            # 多线程处理模式
            pbar_desc = f"[{desc}]: Multi-threaded Tiled Processing" if desc else "Multi-threaded Tiled Processing"
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有tile处理任务
                futures = [executor.submit(process_single_tile, tile_info) for tile_info in indices]
                
                # 使用tqdm显示进度
                pbar = tqdm(as_completed(futures), total=len(futures), desc=pbar_desc, disable=not progress, leave=False)
                
                for future in pbar:
                    result, out_hi, out_hi_end, out_wi, out_wi_end = future.result()
                    out[..., out_hi:out_hi_end, out_wi:out_wi_end] += result
                    count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        else:
            # 原始单线程处理模式
            pbar_desc = f"[{desc}]: Tiled Processing" if desc else "Tiled Processing"
            pbar = tqdm(
                indices, desc=pbar_desc, disable=not progress, leave=False
            )
            for hi, hi_end, wi, wi_end in pbar:
                x_tile = x[..., hi:hi_end, wi:wi_end]
                out_hi, out_hi_end, out_wi, out_wi_end = map(
                    scale_fn, (hi, hi_end, wi, wi_end)
                )
                if len(args) or len(kwargs):
                    kwargs.update(index=TileIndex(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
                out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (
                    fn(x_tile, *args, **kwargs) * weights
                )
                count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        
        out = out / count
        return out

    return tiled_fn


def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def print_vram_state(msg, logger=None):
    alloc = torch.cuda.memory_allocated() / 1024**3
    cache = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if logger:
        logger.info(
            f"[GPU memory]: {msg}, allocated = {alloc:.2f} GB, "
            f"cached = {cache:.2f} GB, peak = {peak:.2f} GB"
        )
    return alloc, cache, peak


class SuppressLogging:

    def __init__(self, level=logging.CRITICAL):
        self.level = level
        self.original_level = logging.getLogger().level

    def __enter__(self):
        logging.getLogger().setLevel(self.level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.original_level)
