from dataclasses import dataclass
from typing import Optional, Tuple, Union, Callable
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as F_t
from transformers.utils import ModelOutput
from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps


@dataclass
class PBRDataClass(ModelOutput):
    r"""
    Properties:
        rough (`torch.Tensor`):
            The roughness tensor of shape `(batch, 1, height, width)`.
        metallic (`torch.Tensor`):
            The metallicness tensor of shape `(batch, 1, height, width)`.
        albedo (`torch.Tensor`):
            The albedo tensor of shape `(batch, 3, height, width)`.
        normal (`torch.Tensor`):
            The normal tensor of shape `(batch, 3, height, width)`.
    """
    pbr: torch.Tensor

    @property
    def rough(self):
        return self.pbr[:, 0:1]

    @property
    def metallic(self):
        return self.pbr[:, 1:2]

    @property
    def albedo(self):
        return self.pbr[:, 2:5]

    @property
    def normal(self):
        return self.pbr[:, 5:8]

    def visualize(self, alpha: torch.Tensor, force_legacy: bool = False):
        # alpha: [batch, 1, height, width]
        assert force_legacy is False, "force_legacy is not supported in public release"
        img_rm = torch.cat([alpha, self.rough, self.metallic, alpha], dim=1)
        img_albedo = torch.cat([self.albedo, alpha], dim=1)
        normal_vec = self.normal * 2 - 1  # [-1, 1]
        normal_vec = normal_vec / (torch.linalg.norm(normal_vec, dim=1, keepdim=True) + 1e-5)
        img_normal = (normal_vec + 1) / 2  # [0, 1]
        img_normal = torch.cat([img_normal, alpha], dim=1)
        return img_rm, img_albedo, img_normal


class PBRReconConfig(PretrainedConfig):
    r"""
    Configuration class for the [`PBRUNetModel`].

    Parameters:
        in_channels (`int`, *optional*, defaults to 4+9*3): Numbers of channels in the input (RGBA) and reference (RGB).
        out_channels (`int`, *optional*, defaults to 1+1+3+3): Number of channels in the PBR output (R, M, A, N).
        center_input_sample (`bool`, *optional*, defaults to `True`): Whether to center the input sample.
        scalar_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of scalar embedding to use.
        num_illumination_params (`int`, *optional*, defaults to 9*2): Number of original illumination parameters (\theta, \phi) under all references.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        downsample_type (`str`, *optional*, defaults to `conv`):
            The downsample type for downsampling layers. Choose between "conv" and "resnet"
        upsample_type (`str`, *optional*, defaults to `conv`):
            The upsample type for upsampling layers. Choose between "conv" and "resnet"
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for normalization.
        attn_norm_num_groups (`int`, *optional*, defaults to `None`):
            If set to an integer, a group norm layer will be created in the mid block's [`Attention`] layer with the
            given number of groups. If left as `None`, the group norm layer will only be created if
            `resnet_time_scale_shift` is set to `default`, and if created will have `norm_num_groups` groups.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"scale_shift"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        loss_weights (`Tuple[float]`, *optional*, defaults to `(1.0, 1.0, 1.0, 1.0)`): Loss weights for L_{R}, L_{M}, L_{A}, L_{N}.
    """

    model_type = "pbr-recon"

    def __init__(
        self,
        in_channels: int = 4+9*3,
        out_channels: int = 1+1+3+3,
        center_input_sample: bool = True,
        scalar_embedding_type: str = "positional",
        num_illumination_params: Optional[int] = 9*2,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "scale_shift",
        add_attention: bool = True,
        num_train_timesteps: Optional[int] = None,
        loss_weights: Tuple[float] = (1.0, 1.0, 1.0, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center_input_sample = center_input_sample
        self.scalar_embedding_type = scalar_embedding_type
        self.num_illumination_params = num_illumination_params
        self.freq_shift = freq_shift
        self.flip_sin_to_cos = flip_sin_to_cos
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.mid_block_scale_factor = mid_block_scale_factor
        self.downsample_padding = downsample_padding
        self.downsample_type = downsample_type
        self.upsample_type = upsample_type
        self.dropout = dropout
        self.act_fn = act_fn
        self.attention_head_dim = attention_head_dim
        self.norm_num_groups = norm_num_groups
        self.attn_norm_num_groups = attn_norm_num_groups
        self.norm_eps = norm_eps
        self.resnet_time_scale_shift = resnet_time_scale_shift
        self.add_attention = add_attention
        self.num_train_timesteps = num_train_timesteps
        self.loss_weights = loss_weights


class PBRUNetModel(PreTrainedModel):
    r"""
    A 2D UNet model that takes an input and optionally several references and outputs PBR images.
    """

    def __init__(
        self,
        config: PBRReconConfig,
    ):
        super().__init__(config)
        self.config: PBRReconConfig = config

        # Check inputs
        if len(config.down_block_types) != len(config.up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {config.down_block_types}. `up_block_types`: {config.up_block_types}."
            )

        if len(config.block_out_channels) != len(config.down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {config.block_out_channels}. `down_block_types`: {config.down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(config.in_channels, config.block_out_channels[0], kernel_size=3, padding=(1, 1))

        # scalar embedding
        if config.scalar_embedding_type == "fourier":
            self.scalar_proj = GaussianFourierProjection(embedding_size=config.block_out_channels[0], scale=16)
            scalar_input_dim = 2 * config.block_out_channels[0]
        elif config.scalar_embedding_type == "positional":
            self.scalar_proj = Timesteps(config.block_out_channels[0], config.flip_sin_to_cos, config.freq_shift)
            scalar_input_dim = config.block_out_channels[0]
        elif config.scalar_embedding_type == "learned":
            self.scalar_proj = nn.Embedding(config.num_train_timesteps, config.block_out_channels[0])
            scalar_input_dim = config.block_out_channels[0]

        # large secondary projection for all reference thetas and phis
        illumination_embed_dim = config.block_out_channels[0] * 4 if config.num_illumination_params is not None else None
        self.illumination_embedding = TimestepEmbedding(scalar_input_dim * config.num_illumination_params, illumination_embed_dim) \
            if config.num_illumination_params is not None else None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = config.block_out_channels[0]
        for i, down_block_type in enumerate(config.down_block_types):
            input_channel = output_channel
            output_channel = config.block_out_channels[i]
            is_final_block = i == len(config.block_out_channels) - 1

            down_block = get_down_block(
            down_block_type,
            num_layers=config.layers_per_block,
            in_channels=input_channel,
            out_channels=output_channel,
            temb_channels=illumination_embed_dim,
            add_downsample=not is_final_block,
            resnet_eps=config.norm_eps,
            resnet_act_fn=config.act_fn,
            resnet_groups=config.norm_num_groups,
            attention_head_dim=config.attention_head_dim if config.attention_head_dim is not None else output_channel,
            downsample_padding=config.downsample_padding,
            resnet_time_scale_shift=config.resnet_time_scale_shift,
            downsample_type=config.downsample_type,
            dropout=config.dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=config.block_out_channels[-1],
            temb_channels=illumination_embed_dim,
            dropout=config.dropout,
            resnet_eps=config.norm_eps,
            resnet_act_fn=config.act_fn,
            output_scale_factor=config.mid_block_scale_factor,
            resnet_time_scale_shift=config.resnet_time_scale_shift,
            attention_head_dim=config.attention_head_dim if config.attention_head_dim is not None else config.block_out_channels[-1],
            resnet_groups=config.norm_num_groups,
            attn_groups=config.attn_norm_num_groups,
            add_attention=config.add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(config.block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(config.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(config.block_out_channels) - 1)]

            is_final_block = i == len(config.block_out_channels) - 1

            up_block = get_up_block(
            up_block_type,
            num_layers=config.layers_per_block + 1,
            in_channels=input_channel,
            out_channels=output_channel,
            prev_output_channel=prev_output_channel,
            temb_channels=illumination_embed_dim,
            add_upsample=not is_final_block,
            resnet_eps=config.norm_eps,
            resnet_act_fn=config.act_fn,
            resnet_groups=config.norm_num_groups,
            attention_head_dim=config.attention_head_dim if config.attention_head_dim is not None else output_channel,
            resnet_time_scale_shift=config.resnet_time_scale_shift,
            upsample_type=config.upsample_type,
            dropout=config.dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = config.norm_num_groups if config.norm_num_groups is not None else min(config.block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=config.block_out_channels[0], num_groups=num_groups_out, eps=config.norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(config.block_out_channels[0], config.out_channels, kernel_size=3, padding=1)

        # transformers post init
        self.post_init()

    def forward(
        self,
        sample: torch.Tensor,
        illumination: Optional[torch.Tensor],
    ) -> Union[PBRDataClass, Tuple]:
        r"""
        The [`PBRUNetModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The unet input tensor with the following shape `(batch, channel, height, width)`.
            illumination (`torch.Tensor`): The (theta, phi) pairs for all references of shape `(batch, N_ref, 2)`.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. illumination
        if self.illumination_embedding is not None:
            i_embs = self.scalar_proj(illumination.reshape(-1)).reshape(illumination.shape[0], -1)

            # i_embs does not contain any weights and will always return f32 tensors
            # but illumination_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            i_embs = i_embs.to(dtype=self.dtype)
            emb = self.illumination_embedding(i_embs)
        else:
            emb = None

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.scalar_embedding_type == "fourier":
            raise NotImplementedError("Fourier embedding is not supported for PBRUNetModel.")
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        return PBRDataClass(pbr=sample)


class PBRUNetModelForReconstruction(PBRUNetModel):
    def __init__(self, config: PBRReconConfig):
        super().__init__(config)
        self.loss_fn = nn.MSELoss()
        assert sum(config.loss_weights) == 4.0, f"Loss weights scale unreasonable: sum({config.loss_weights}) != 4.0"
        if len(config.loss_weights) == 4:
            print(f"Receiving 4 loss weights, appending 0.0 for normal L2 loss.")
            config.loss_weights = (*config.loss_weights, 0.0)
        self.post_init()

    def pack_unet_input_sample(self, input_image: torch.Tensor, reference_images: Optional[torch.Tensor]) -> torch.Tensor:
        if reference_images is None:
            return input_image
        resized_reference_images = torch.stack([
            F_t.resize(ref_img, size=input_image.shape[-2:], interpolation=F_t.InterpolationMode.BICUBIC).clamp(0, 1)
            for ref_img in reference_images.unbind(dim=1)
        ], dim=1)  # BxNx3xHxW
        reference_images = torch.flatten(resized_reference_images, start_dim=1, end_dim=2)  # Bx(Nx3)xHxW
        packed_sample = torch.cat([input_image, reference_images], dim=1)  # Bx(4+Nx3)xHxW
        return packed_sample

    @staticmethod
    def cut_ref_images(ref_image):
        B, C, H_3, W_3 = ref_image.shape
        assert H_3 % 3 == 0 and W_3 % 3 == 0, "Height and Width must be divisible by 3."
        H, W = H_3 // 3, W_3 // 3
        ref_image = ref_image.view(B, C, 3, H, 3, W)
        ref_image = ref_image.permute(0, 2, 4, 1, 3, 5)
        ref_image = ref_image.reshape(B, 9, C, H, W)
        return ref_image

    def predict(self, input_image: Image, ref_image: Image) -> Image:
        # make sure input_image is in desired resolution
        input_image = F_t.to_tensor(input_image).unsqueeze(0).to(self.device)
        ref_image = F_t.to_tensor(ref_image).unsqueeze(0).to(self.device)
        ref_image = self.cut_ref_images(ref_image)
        # pack
        packed_sample = self.pack_unet_input_sample(input_image, ref_image)
        # forward
        thetas = torch.arange(9) * torch.pi / 4
        phis = torch.tensor([1., 2., 1., 2., 1., 2., 1., 2., 0]) * torch.pi / 6
        standard_illumination = torch.stack([thetas, phis], dim=1).unsqueeze(0).repeat(input_image.shape[0], 1, 1).to(self.device)
        # call parent class forward
        output = super().forward(sample=packed_sample, illumination=standard_illumination)
        # visualize
        img_rm, img_albedo, img_normal = output.visualize(alpha=input_image[:, 3:])
        # clamp (0,1)
        img_rm = torch.clamp(img_rm, 0, 1)
        img_albedo = torch.clamp(img_albedo, 0, 1)
        img_normal = torch.clamp(img_normal, 0, 1)
        img_rm = F_t.to_pil_image(img_rm[0])
        img_albedo = F_t.to_pil_image(img_albedo[0])
        img_normal = F_t.to_pil_image(img_normal[0])
        return img_rm, img_albedo, img_normal

    def forward(
        self,
        uid: list[str],
        input_image: torch.Tensor,
        reference_images: Optional[torch.Tensor],
        pbr_image: torch.Tensor,
        illumination: Optional[torch.Tensor],
        return_dict: bool = True,
        return_loss: bool = True,
        return_out_gt: bool = False,
        **kwargs,
    ) -> Union[ModelOutput, Tuple]:
        del kwargs

        sample = self.pack_unet_input_sample(input_image, reference_images)
        output = super().forward(sample=sample, illumination=illumination)
        target = PBRDataClass(pbr=pbr_image)

        # MSE loss for each within [0, 1] value range
        loss_rough = self.loss_fn(output.rough, target.rough)
        loss_metallic = self.loss_fn(output.metallic, target.metallic)
        loss_albedo = self.loss_fn(output.albedo, target.albedo)
        loss_normal_l2 = self.loss_fn(output.normal, target.normal)

        # Calculate similarity of normalized unit vectors for normal loss
        output_normal_vec = output.normal * 2 - 1  # [-1, 1]
        target_normal_vec = target.normal * 2 - 1  # [-1, 1]
        output_normal_dir = output_normal_vec / (torch.linalg.norm(output_normal_vec, dim=1, keepdim=True) + 1e-5)
        target_normal_dir = target_normal_vec / (torch.linalg.norm(target_normal_vec, dim=1, keepdim=True) + 1e-5)
        loss_normal_cos = 1 - torch.sum(output_normal_dir * target_normal_dir, dim=1)

        loss = sum(loss_value * loss_weight for loss_value, loss_weight in \
            zip([loss_rough, loss_metallic, loss_albedo, loss_normal_cos, loss_normal_l2], self.config.loss_weights))

        assert return_dict, "Only return_dict=True is supported."
        assert return_loss, "Only return_loss=True is supported."
        return ModelOutput(
            loss=loss.mean(),
            losses=ModelOutput(
                loss_rough=loss_rough.mean(),
                loss_metallic=loss_metallic.mean(),
                loss_albedo=loss_albedo.mean(),
                loss_normal_cos=loss_normal_cos.mean(),
                loss_normal_l2=loss_normal_l2.mean(),
            ),
            **({"uid": uid, "unet_out": output, "target": target} if return_out_gt else {}),
        )


def pbr_recon_model_init_factory(load_from: Optional[str] = None, **model_args) -> Callable[[], PBRUNetModelForReconstruction]:
    def pbr_recon_model_init() -> PBRUNetModelForReconstruction:
        config = PBRReconConfig(**model_args)
        if load_from is not None:
            print("Loading from checkpoint:", load_from)
            model = PBRUNetModelForReconstruction.from_pretrained(load_from, config=config)
        else:
            model = PBRUNetModelForReconstruction(config)
        return model
    return pbr_recon_model_init
