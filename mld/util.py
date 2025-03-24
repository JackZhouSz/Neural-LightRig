import importlib
from PIL import Image
import numpy as np
import torch


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def get_obj_from_str(string: str, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def set_grad(model: torch.nn.Module, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


@torch.no_grad()
def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def get_auto_ema_decay(
    steps: int,
    update_after_step: int = 0,
    use_ema_warmup: bool = False,
    inv_gamma: float = 1.0,
    power: float = 2 / 3,
    max_decay: float = 0.99,
    min_decay: float = 0.0,
    ):
    step = max(0, steps - update_after_step)
    if step <= 0:
        return 0.0
    if use_ema_warmup:
        cur_decay_value = 1 - (1 + step / inv_gamma) ** -power
    else:
        cur_decay_value = (1 + step) / (10 + step)
    cur_decay_value = min(cur_decay_value, max_decay)
    cur_decay_value = max(cur_decay_value, min_decay)
    return cur_decay_value
