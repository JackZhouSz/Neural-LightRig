from copy import deepcopy
from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import transformers
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available

from .util import to_rgb_image, set_grad
from .util import scale_latents, unscale_latents, scale_image, unscale_image


class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask)
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class ElasticConv2d(nn.Conv2d):
    def __init__(self, pretrained_conv: nn.Conv2d) -> None:
        super().__init__(
            pretrained_conv.in_channels * 2, pretrained_conv.out_channels,
            kernel_size=pretrained_conv.kernel_size, stride=pretrained_conv.stride,
            padding=pretrained_conv.padding, padding_mode=pretrained_conv.padding_mode,
            dilation=pretrained_conv.dilation, groups=pretrained_conv.groups, bias=pretrained_conv.bias is not None,
            device=pretrained_conv.weight.device, dtype=pretrained_conv.weight.dtype,
        )
        with torch.no_grad():
            self.weight[:, :pretrained_conv.in_channels] = pretrained_conv.weight
            if pretrained_conv.bias is not None:
                self.bias = nn.Parameter(pretrained_conv.bias.clone())
        self._register_load_state_dict_pre_hook(self.elastic_conv_pre_hook)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[1] == self.in_channels:
            return self._conv_forward(input, self.weight, self.bias)
        elif input.shape[1] == self.in_channels // 2:
            return self._conv_forward(input, self.weight[:, :input.shape[1]], self.bias)
        else:
            raise ValueError(f"Input shape {input.shape} does not match in_channels {self.in_channels} or half of it.")

    def elastic_conv_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        A pre-hook that handles elastic convolutional layers where part of the weights come from the
        state_dict (checkpoint) and part are left as initialized in the model.
        """
        # Get the key for the layer's weights in the state_dict (e.g., 'conv_in.weight')
        conv_key = prefix + 'weight'
        
        if conv_key in state_dict:
            pretrained_weight = state_dict[conv_key]

            # Check if the pretrained model has fewer input channels (e.g., 4 instead of 8)
            if pretrained_weight.shape[1] < self.weight.shape[1]:
                print(f"Elastic loading: Adapting {conv_key} weights from {pretrained_weight.shape[1]} channels to {self.weight.shape[1]} channels.")

                # Create a new tensor with the model's current shape, keeping its default initialization
                new_weight = self.weight.clone()

                # Copy the existing pretrained weights to the first part of the new weight tensor
                new_weight[:, :pretrained_weight.shape[1], :, :] = pretrained_weight

                # Assign the new weight tensor to the state_dict
                state_dict[conv_key] = new_weight


class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler,
        enable_concats: bool,
        disable_rca: bool,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched
        self.enable_concats = enable_concats
        self.disable_rca = disable_rca

        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, name=name,
                enabled=name.endswith("attn1.processor") and not disable_rca,
            )
        unet.set_attn_processor(unet_lora_attn_procs)

        if self.enable_concats:
            unet.conv_in = ElasticConv2d(unet.conv_in)
            print(f"Switched unet.conv_in to ElasticConv2d")
        self._registered_concat_latents = None

    @property
    def registered_concat_latents(self):
        if not self.enable_concats:
            raise ValueError("Querying concat latents when enable_concats is False.")
        return self._registered_concat_latents

    @registered_concat_latents.setter
    def registered_concat_latents(self, value):
        if not self.enable_concats:
            raise ValueError("Registering concat latents when enable_concats is False.")
        self._registered_concat_latents = value

    @registered_concat_latents.deleter
    def registered_concat_latents(self):
        if not self.enable_concats:
            raise ValueError("Deleting concat latents when enable_concats is False.")
        self._registered_concat_latents = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        **kwargs
    ):
        cond_lat = cross_attention_kwargs['cond_lat']
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = torch.randn_like(cond_lat)
        if self.training:
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)
        else:
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, timestep.reshape(-1))
        ref_dict = {}
        if not self.disable_rca:
            self.forward_cond(
                noisy_cond_lat, timestep,
                encoder_hidden_states, class_labels,
                ref_dict, is_cfg_guidance, **kwargs
            )
        weight_dtype = self.unet.dtype
        if self.enable_concats and self.registered_concat_latents is not None:
            # this code block should be triggered only when calling pipeline directly
            assert sample.shape[0] in [1, 2], f"Sample shape {sample.shape} not supported when using concat register"
            concats = self.registered_concat_latents.repeat(1, 1, 3, 3)
            if sample.shape[0] == 2:
                # concats should be zeros for unconditional latents to support cfg
                concats = torch.cat([torch.zeros_like(concats), concats], dim=0)
            sample = torch.cat([sample, concats], dim=1)
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


class ModuleListDict(torch.nn.Module):
    def __init__(self, procs: dict) -> None:
        super().__init__()
        self.keys = sorted(procs.keys())
        self.values = torch.nn.ModuleList(procs[k] for k in self.keys)

    def __getitem__(self, key):
        return self.values[self.keys.index(key)]


class MultiLightDiffusionPipeline(diffusers.StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: RefOnlyNoisedUNet
    ema: Optional[RefOnlyNoisedUNet]
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping_weights: nn.Parameter

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: RefOnlyNoisedUNet,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: Optional[transformers.CLIPVisionModelWithProjection] = None,
        feature_extractor_clip: Optional[CLIPImageProcessor] = None,
        feature_extractor_vae: Optional[CLIPImageProcessor] = None,
        ramping_weights: Optional[nn.Parameter] = None,
        store_ema: bool = False,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae,
            ramping_weights=ramping_weights,
        )
        self.register_to_config(store_ema=store_ema)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if store_ema:
            self.ema = deepcopy(unet).eval()
            set_grad(self.ema, False)
        else:
            self.ema = None

        set_grad(self.vae, False)
        set_grad(self.text_encoder, False)
        set_grad(self.vision_encoder, False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.vae = self.vae.to(*args, **kwargs)
        self.text_encoder = self.text_encoder.to(*args, **kwargs)
        self.unet = self.unet.to(*args, **kwargs)
        self.vision_encoder = self.vision_encoder.to(*args, **kwargs)
        self.ramping_weights = self.ramping_weights.to(*args, **kwargs)
        if self.ema is not None:
            self.ema = self.ema.to(*args, **kwargs)
        return self

    def prepare(self):
        train_sched = DDPMScheduler.from_config(self.scheduler.config)
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler).eval()

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image

    def encode_concat_image(self, image: Image, under_resolution: int = 256):
        dtype = next(self.vae.parameters()).dtype
        device = self.vae.device
        images = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.
        images = v2.functional.resize(images, under_resolution, interpolation=3, antialias=True).clamp(0, 1)
        images = (images - 0.5) / 0.8  # [-0.625, 0.625]
        latents = self.vae.encode(images.to(device, dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image = None,
        prompt = "",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=2.0,
        guidance_rescale=0.7,
        output_type: Optional[str] = "pil",
        width=768,
        height=768,
        num_inference_steps=75,
        return_dict=True,
        use_ema: bool = False,  # NOTE: public version does not support EMA
        **kwargs
    ):
        self.prepare()
        if image is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(image, torch.Tensor)
        image = to_rgb_image(image)
        image_0 = self.feature_extractor_vae(images=to_rgb_image(Image.new("RGB", image.size, color=(0, 0, 0))), return_tensors="pt").pixel_values
        image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
        image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values

        if use_ema and self.ema is None:
            print(f"Warning: EMA is not available for this pipeline. Falling back to the main model.")
            use_ema = False
        if use_ema:
            self.unet, self.ema = self.ema, self.unet
        model = self.unet

        if model.enable_concats:
            model.registered_concat_latents = self.encode_concat_image(image)
        image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
        image_0 = image_0.to(device=self.vae.device, dtype=self.vae.dtype)
        cond_lat = self.encode_condition_image(image)
        if guidance_scale > 1:
            negative_lat = self.encode_condition_image(image_0)
            cond_lat = torch.cat([negative_lat, cond_lat])
        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)
        
        if hasattr(self, "encode_prompt"):
            encoder_hidden_states = self.encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )[0]
        else:
            encoder_hidden_states = self._encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )
        ramp = self.ramping_weights.unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        cak = dict(cond_lat=cond_lat)
        latents: torch.Tensor = super().__call__(
            None,
            *args,
            cross_attention_kwargs=cak,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            output_type='latent',
            width=width,
            height=height,
            **kwargs
        ).images
        latents = unscale_latents(latents)
        if not output_type == "latent":
            image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        else:
            image = latents

        if model.enable_concats:
            del model.registered_concat_latents

        if use_ema:
            self.unet, self.ema = self.ema, self.unet

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


def multi_light_diff_pipeline_init(enable_concats: bool, disable_rca: bool, store_ema: bool, sd_21_pretrain: Optional[str]):
    DEFAULT_PRETRAIN = "stabilityai/stable-diffusion-2-1-unclip"
    if sd_21_pretrain is None:
        sd_21_pretrain = DEFAULT_PRETRAIN
    sd_21_pipe = DiffusionPipeline.from_pretrained(sd_21_pretrain)
    train_sched = DDPMScheduler.from_config('sudo-ai/zero123plus-v1.2', subfolder='scheduler', timestep_spacing='trailing')
    infer_sched = EulerAncestralDiscreteScheduler.from_config(train_sched.config, rescale_betas_zero_snr=True)
    vae = sd_21_pipe.vae
    unet = UNet2DConditionModel.from_pretrained(sd_21_pretrain, subfolder='unet', class_embed_type=None)
    unet = RefOnlyNoisedUNet(unet, train_sched, infer_sched, enable_concats=enable_concats, disable_rca=disable_rca)
    text_encoder = sd_21_pipe.text_encoder
    tokenizer = sd_21_pipe.tokenizer
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", subfolder='image_encoder')
    feature_extractor_clip = sd_21_pipe.feature_extractor
    feature_extractor_vae = CLIPImageProcessor.from_pretrained("sudo-ai/zero123plus-v1.2", subfolder='feature_extractor_vae')
    print(f"Initializing ramping weights with linear guidance from FlexDiffuse.")
    ramping_coefficients = torch.linspace(0.0, 1.0, text_encoder.config.max_position_embeddings+1).tolist()[1:]
    ramping_weights = nn.Parameter(torch.tensor(ramping_coefficients))
    pipeline = MultiLightDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=infer_sched,
        vision_encoder=vision_encoder,
        feature_extractor_clip=feature_extractor_clip,
        feature_extractor_vae=feature_extractor_vae,
        ramping_weights=ramping_weights,
        store_ema=store_ema,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, train_sched
