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
    
    def on_fit_start(self):
        self.pipeline = self.pipeline.to(self.device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)

    def on_before_zero_grad(self, optimizer):
        if self.ema_decay > 1.:
            # using automatic ema decay
            exec_ema_decay = get_auto_ema_decay(steps=self.global_step)
        elif self.ema_decay > 0:
            # fixed ema decay
            exec_ema_decay = self.ema_decay
        else:
            # no ema
            return
        update_ema(self.unet, self.ema, exec_ema_decay)
    
    def prepare_batch_data(self, batch):
        # prepare stable diffusion input
        cond_imgs = batch['cond_imgs']      # (B, C, H, W)
        cond_imgs = cond_imgs.to(self.device)

        # random resize the condition image at training, TODO: hard coded
        if self.training:
            cond_size = np.random.randint(256, 513)
        else:
            cond_size = 512
        cond_imgs = v2.functional.resize(cond_imgs, cond_size, interpolation=3, antialias=True).clamp(0, 1)

        _unet_resolution = 256  # TODO: hard coded resolution
        concat_imgs = v2.functional.resize(cond_imgs, _unet_resolution, interpolation=3, antialias=True).clamp(0, 1)

        target_imgs = batch['target_imgs']  # (B, 9, C, H, W)
        target_imgs = v2.functional.resize(target_imgs, _unet_resolution, interpolation=3, antialias=True).clamp(0, 1)
        target_imgs = rearrange(target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=3)    # (B, C, 3H, 3W)
        target_imgs = target_imgs.to(self.device)

        return cond_imgs, target_imgs, concat_imgs
    
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
    
    @torch.no_grad()
    def encode_condition_image(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
        image_pt = self.pipeline.feature_extractor_vae(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        latents = self.pipeline.vae.encode(image_pt).latent_dist.sample()
        return latents
    
    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8   # [-0.625, 0.625]
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents
    
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
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    def training_step(self, batch, batch_idx):
        # get input
        cond_imgs, target_imgs, concat_imgs = self.prepare_batch_data(batch)

        # sample random timestep
        B = cond_imgs.shape[0]
        
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            prompt_embeds = self.pipeline.encode_prompt([""]*B, self.device, 1, False)[0]
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))
            concat_latents = self.encode_target_images(torch.zeros_like(concat_imgs))  # resuing target encoding for concat
        else:
            prompt_embeds = self.forward_vision_encoder(cond_imgs)
            cond_latents = self.encode_condition_image(cond_imgs)
            concat_latents = self.encode_target_images(concat_imgs)  # resuing target encoding for concat

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)

        if self.enable_concats:
            input_latent = torch.cat([latents_noisy, concat_latents.repeat(1, 1, 3, 3)], dim=1)
        else:
            input_latent = latents_noisy

        v_pred = self.forward_unet(input_latent, t, prompt_embeds, cond_latents)
        v_target = self.get_v(latents, noise, t)

        loss, loss_dict = self.compute_loss(v_pred, v_target)

        # logging
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.global_step % 500 == 0 and self.global_rank == 0:
            with torch.no_grad():
                latents_pred = self.predict_start_from_z_and_v(latents_noisy, t, v_pred)

                latents = unscale_latents(latents_pred)
                images = unscale_image(self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
                images = (images * 0.5 + 0.5).clamp(0, 1)
                images = torch.cat([target_imgs, images], dim=-2)

                save_image(
                    images, os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png'),
                    nrow=images.shape[0], normalize=True, value_range=(0, 1),
                )

        return loss
        
    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if len(self.validation_step_outputs) >= self.n_val_batches:
            return

        # get input
        cond_imgs, target_imgs, concat_imgs = self.prepare_batch_data(batch)
        images_pil = [v2.functional.to_pil_image(cond_imgs[i]) for i in range(cond_imgs.shape[0])]

        use_ema_pool = [False]
        if self.ema_decay > 0:
            use_ema_pool.append(True)

        for use_ema in use_ema_pool:
            outputs = []
            for cond_img in images_pil:
                latent = self.pipeline(
                    image=cond_img,
                    num_inference_steps=75,
                    guidance_scale=2.0,
                    guidance_rescale=0.7,
                    use_ema=use_ema,
                ).images
                outputs.append(v2.functional.to_tensor(latent[0]))
            outputs = torch.stack(outputs, dim=0).to(self.device)
            images = torch.cat([target_imgs, outputs], dim=-2)
            if use_ema:
                self.validation_step_outputs_ema.append(images)
            else:
                self.validation_step_outputs.append(images)
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        val_outputs_pool = [self.validation_step_outputs]
        use_ema_pool = [False]
        if self.validation_step_outputs_ema:
            val_outputs_pool.append(self.validation_step_outputs_ema)
            use_ema_pool.append(True)

        for use_ema, val_outputs in zip(use_ema_pool, val_outputs_pool):
            images = torch.cat(val_outputs, dim=0)

            all_images = self.all_gather(images)
            all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

            if all_images.shape[0] > self.max_val_samples:
                all_images = all_images[:self.max_val_samples]

            if self.global_rank == 0:
                save_image(
                    all_images, os.path.join(self.logdir, 'images_val', f"val_{self.global_step:07d}{'_ema' if use_ema else ''}.png"),
                    nrow=8, normalize=True, value_range=(0, 1),
                )

        self.validation_step_outputs.clear()
        self.validation_step_outputs_ema.clear()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if hasattr(self.trainer.train_dataloader, 'with_length'):
            self.trainer.train_dataloader.with_length(self.trainer.max_steps - self.trainer.global_step)

    def configure_optimizers(self):
        lr = self.train_cfg.learning_rate
        warmup_steps = self.train_cfg.warmup_steps
        max_steps = self.train_cfg.max_steps

        if self.tuning_strategy.startswith('sa+cakv'):
            set_grad(self.unet, False)
            for name, module in self.unet.named_modules():
                if isinstance(module, Attention):
                    if 'attn1' in name:
                        set_grad(module, True)
                    elif 'attn2' in name:
                        set_grad(module.to_k, True)
                        set_grad(module.to_v, True)
                    else:
                        raise ValueError(f"Unexpected module name: {name}")
            if 'inconv' in self.tuning_strategy:
                set_grad(self.unet.conv_in, True)
        elif self.tuning_strategy == 'full':
            set_grad(self.unet, True)
        else:
            raise ValueError(f"Unexpected tuning strategy: {self.tuning_strategy}")

        params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters())) + [self.ramping_weights]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
