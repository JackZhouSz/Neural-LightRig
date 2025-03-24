import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.transforms import v2
from torchvision.utils import save_image
from einops import rearrange
from diffusers.models.attention_processor import Attention
from transformers import get_cosine_schedule_with_warmup

from .pipeline import multi_light_diff_pipeline_init
from .util import set_grad, update_ema, get_auto_ema_decay
from .util import scale_latents, unscale_latents, scale_image, unscale_image
from .util import extract_into_tensor


class MVDiffusion(pl.LightningModule):
    def __init__(
        self,
        train_cfg,
        drop_cond_prob=0.1,
        tuning_strategy='sa+cakv',  # 'full'
        enable_concats=False,
        disable_rca=False,
        n_val_batches=8,
        max_val_samples=32,
        ema_decay=-1,
        sd_21_pretrain=None,
    ):
        super(MVDiffusion, self).__init__()

        self.train_cfg = train_cfg
        self.drop_cond_prob = drop_cond_prob
        self.tuning_strategy = tuning_strategy
        self.n_val_batches = n_val_batches
        self.max_val_samples = max_val_samples
        self.ema_decay = ema_decay

        self.register_schedule()

        if 'inconv' in tuning_strategy and not enable_concats:
            print("Warning: 'inconv' tuning strategy is set while enable_concats is False.")
        self.enable_concats = enable_concats
        self.disable_rca = disable_rca

        # init modules
        self.pipeline, self.train_scheduler = multi_light_diff_pipeline_init(
            enable_concats=self.enable_concats,
            disable_rca=disable_rca,
            store_ema=(self.ema_decay > 0),
            sd_21_pretrain=sd_21_pretrain,
        )
        self.unet = self.pipeline.unet
        self.ramping_weights = self.pipeline.ramping_weights
        self.ema = self.pipeline.ema
        if self.ema_decay > 0:
            self._register_load_state_dict_pre_hook(self._ema_load_state_dict_pre_hook)

        # validation output buffer
        self.validation_step_outputs = []
        self.validation_step_outputs_ema = []

    def _ema_load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """This hook will copy the unet's weights to ema if ema is missing from the checkpoint."""
        
        # Look for the presence of 'ema' keys in the state dict
        ema_key_prefix = 'ema'
        unet_key_prefix = 'unet'

        # Check if any ema keys exist in the state_dict
        ema_keys_exist = any(key.startswith(ema_key_prefix) for key in state_dict.keys())
        
        if not ema_keys_exist:
            print(f"Warning: EMA weights not found in the checkpoint. Copying '{unet_key_prefix}' weights to '{ema_key_prefix}'.")

            # Copy the unet weights to ema
            for key in list(state_dict.keys()):
                if key.startswith(unet_key_prefix):
                    # Replace 'unet' with 'ema' in the key
                    ema_key = key.replace(unet_key_prefix, ema_key_prefix, 1)  # Replace only the first occurrence of 'unet'
                    state_dict[ema_key] = state_dict[key]  # Copy the unet weight to ema weight

    def register_schedule(self):
        self.num_timesteps = 1000

        # replace scaled_linear schedule with linear schedule as Zero123++
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
        
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())
    
    def forward_vision_encoder(self, images):
        # make sure ramping weights calculate gradients
        with torch.no_grad():
            dtype = next(self.pipeline.vision_encoder.parameters()).dtype
            image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
            image_pt = self.pipeline.feature_extractor_clip(images=image_pil, return_tensors="pt").pixel_values
            image_pt = image_pt.to(device=self.device, dtype=dtype)
            global_embeds = self.pipeline.vision_encoder(image_pt, output_hidden_states=False).image_embeds
            global_embeds = global_embeds.unsqueeze(-2)
            encoder_hidden_states = self.pipeline.encode_prompt("", self.device, 1, False)[0]

        # critical modification to include self.ramping_weights in gradient computation
        ramp = self.ramping_weights.unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

        return encoder_hidden_states
    
    def forward_unet(self, latents, t, prompt_embeds, cond_latents):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        pred_noise = self.pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        return pred_noise
