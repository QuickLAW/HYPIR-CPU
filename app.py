import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="configs/sd2_gradio.yaml")
parser.add_argument("--local", action="store_true")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--gpt_caption", action="store_true")
parser.add_argument("--max_size", type=str, default=None, help="Comma-seperated image size")
parser.add_argument("--device", type=str)
args = parser.parse_args()

import random
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from dotenv import load_dotenv
from PIL import Image
import gradio as gr
import torchvision.transforms as transforms

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import GPTCaptioner

# 设置环境变量，禁用所有检查
os.environ['GRADIO_DEBUG'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

load_dotenv()
error_image = Image.open(os.path.join("assets", "gradio_error_img.png"))

if args.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AssertionError("Torch not compiled with CUDA enabled")

# CPU性能优化设置
if device == "cpu":
    from HYPIR.utils.device_setup import setup_cpu_device
    setup_cpu_device()
    try:
        torch.set_num_threads(8)
        print(f"  - PyTorch线程数: {torch.get_num_threads()}")
    except RuntimeError as e:
        print(f"  - 线程数设置跳过: {e}")
    torch.backends.mkldnn.enabled = True
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    print("CPU优化设置已应用:")
    print(f"  - MKL-DNN加速: {torch.backends.mkldnn.enabled}")
    print(f"  - OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  - MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")

max_size = args.max_size
if max_size is not None:
    max_size = tuple(int(x) for x in max_size.split(","))
    if len(max_size) != 2:
        raise ValueError(f"Invalid max size: {max_size}")
    print(f"Max size set to {max_size}, max pixels: {max_size[0] * max_size[1]}")

if args.gpt_caption:
    if (
        "GPT_API_KEY" not in os.environ
        or "GPT_BASE_URL" not in os.environ
        or "GPT_MODEL" not in os.environ
    ):
        raise ValueError(
            "If you want to use gpt-generated caption, "
            "please specify both `GPT_API_KEY`, `GPT_BASE_URL` and `GPT_MODEL` in your .env file. "
            "See README.md for more details."
        )
    captioner = GPTCaptioner(
        api_key=os.getenv("GPT_API_KEY"),
        base_url=os.getenv("GPT_BASE_URL"),
        model=os.getenv("GPT_MODEL"),
    )
to_tensor = transforms.ToTensor()

config = OmegaConf.load(args.config)
if config.base_model_type == "sd2":
    model = SD2Enhancer(
        base_model_path=config.base_model_path,
        weight_path=config.weight_path,
        lora_modules=config.lora_modules,
        lora_rank=config.lora_rank,
        model_t=config.model_t,
        coeff_t=config.coeff_t,
        device=device,
    )
    model.init_models()
else:
    raise ValueError(config.base_model_type)


def process(
    image,
    prompt,
    upscale,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
    image = image.convert("RGB")
    # Check image size
    if max_size is not None:
        out_w, out_h = tuple(int(x * upscale) for x in image.size)
        if out_w * out_h > max_size[0] * max_size[1]:
            return error_image, (
                "Failed: The requested resolution exceeds the maximum pixel limit. "
                f"Your requested resolution is ({out_h}, {out_w}). "
                f"The maximum allowed pixel count is {max_size[0]} x {max_size[1]} "
                f"= {max_size[0] * max_size[1]} :("
            )
    if prompt == "auto":
        if args.gpt_caption:
            prompt = captioner(image)
        else:
            return error_image, "Failed: This gradio is not launched with gpt-caption support :("

    image_tensor = to_tensor(image).unsqueeze(0)
    try:
        pil_image = model.enhance(
            lq=image_tensor,
            prompt=prompt,
            upscale=upscale,
            return_type="pil",
        )[0]
    except Exception as e:
        return error_image, f"Failed: {e} :("

    return pil_image, f"Success! :)\nUsed prompt: {prompt}"


MARKDOWN = """
## HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration

[GitHub](https://github.com/XPixelGroup/HYPIR) | [Paper](TODO) | [Project Page](TODO)

If HYPIR is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            prompt = gr.Textbox(label=(
                "Prompt (Input 'auto' to use gpt-generated caption)"
                if args.gpt_caption else "Prompt"
            ))
            upscale = gr.Slider(minimum=1, maximum=8, value=1, label="Upscale Factor", step=1)
            seed = gr.Number(label="Seed", value=-1)
            run = gr.Button(value="Run")
        with gr.Column():
            result = gr.Image(type="pil", format="png")
            status = gr.Textbox(label="status", interactive=False)
        run.click(
            fn=process,
            inputs=[image, prompt, upscale, seed],
            outputs=[result, status],
        )

# 使用更宽松的启动参数
try:
    block.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        inbrowser=False,
        prevent_thread_lock=False,
        quiet=False,
        show_error=True,
        ssl_verify=False
    )
except Exception as e:
    print(f"Launch with queue failed: {e}")
    print("Trying to launch without queue...")
    block.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        inbrowser=False,
        prevent_thread_lock=False,
        quiet=False,
        show_error=True
    )
