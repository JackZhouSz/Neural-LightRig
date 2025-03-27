
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers import TrainerCallback, TrainingArguments
from transformers.integrations.integration_utils import WandbCallback


def create_grid(images, grid_size=(3, 3), resolution = None):
    """
    Create a grid of images. Assumes images are in the format BxNxCxHxW.
    grid_size should be (rows, cols).
    """
    B, N, C, H, W = images.shape
    assert N == grid_size[0] * grid_size[1], "Number of images must match grid dimensions"
    images = images.view(B, grid_size[0], grid_size[1], C, H, W)  # BxRowxColxCxHxW
    images = images.permute(0, 3, 1, 4, 2, 5)  # BxCxRowxHxColxW
    images = images.contiguous().view(B, C, grid_size[0] * H, grid_size[1] * W)  # BxCx(3*H)x(3*W)
    if resolution is not None:
        images = transforms.functional.resize(images, size=resolution)
    return images


class VisualizePBRReconCallback(TrainerCallback):
    def __init__(self, n_visualize_samples: int = 8, dump_prefix: str = 'visualize', on_rank_zero_only: bool = True):
        self.n_visualize_samples = n_visualize_samples
        self.dump_prefix = dump_prefix
        self.on_rank_zero_only = on_rank_zero_only

    def _save_images(self, img_torch, dump_dir, prefix, global_step, local_rank, vis_uids):
        """Helper method to save visualized images and uids."""
        img_torch = img_torch.flatten(0, 1)  # (BxH)x(6xW)xC
        img_torch = img_torch.clamp(0, 1)
        img_np = img_torch.cpu().numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save(os.path.join(dump_dir, f'{prefix}_{global_step:06d}_{local_rank}.png'))
        
        with open(os.path.join(dump_dir, f'{prefix}_{global_step:06d}_{local_rank}.txt'), 'w') as f:
            for uid in vis_uids:
                f.write(f'{uid}\n')

    def _visualize_step(self, model, batch):
        """Common method to visualize a single batch."""
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                predictions = model(**batch, return_out_gt=True)
        img_in = batch['input_image']  # BxCxHxW
        if batch.get('reference_images') is not None:
            img_ref = torch.stack([
                transforms.functional.resize(_img, size=img_in.shape[-2:]) for _img in batch['reference_images'].unbind(dim=1)
            ], dim=1)  # BxNx3xHxW
            img_ref = torch.cat([img_ref, img_in[:, 3:].repeat(1, img_ref.shape[1], 1, 1).unsqueeze(2)], dim=2)  # BxNxCxHxW
            if img_ref.shape[1] < 9:
                img_ref = torch.cat([img_ref, torch.zeros_like(img_ref[:, :1]).repeat(1, 9 - img_ref.shape[1], 1, 1, 1)], dim=1)
            assert img_ref.shape[1] == 9, "Expected 9 reference images!"
            img_ref = create_grid(img_ref, grid_size=(3, 3), resolution=img_in.shape[-2:])  # BxCx(nh)x(nh) <--> BxCxHxW
            img_ref = (img_ref,)
        else:
            img_ref = ()
        img_rm_out, img_albedo_out, img_normal_out = predictions['unet_out'].visualize(batch['input_image'][:, 3:])  # each BxCxHxW
        img_rm_gt, img_albedo_gt, img_normal_gt = predictions['target'].visualize(batch['input_image'][:, 3:])
        img = torch.cat([img_in, *img_ref, img_rm_out, img_rm_gt, img_albedo_out, img_albedo_gt, img_normal_out, img_normal_gt], dim=3)  # BxCxHx(7xW) or BxCxHx(8xW)
        img = img.permute(0, 2, 3, 1)  # BxHx(7xW)xC or BxHx(8xW)xC
        return img

    def _visualize_loop(self, model, dataloader, dump_dir, prefix, global_step, local_rank):
        """Shared method to handle both training and evaluation visualization."""
        vis_uids = []
        img_torch = None
        imgaug_torch = None

        was_training = model.training
        model.eval()

        for batch in dataloader:
            if img_torch is not None and img_torch.shape[0] >= self.n_visualize_samples:
                break
            vis_uids.extend(batch['uid'])
            img = self._visualize_step(model, batch)
            img_torch = torch.cat([img_torch, img], dim=0) if img_torch is not None else img
            if batch.get('reference_images_raw') is not None or batch.get('illumination_raw') is not None:
                print("Raw visualization samples found!")
                if batch.get('reference_images_raw') is not None:
                    batch['reference_images'] = batch['reference_images_raw']
                if batch.get('illumination_raw') is not None:
                    batch['illumination'] = batch['illumination_raw']
                imgaug = self._visualize_step(model, batch)
                imgaug_torch = torch.cat([imgaug_torch, imgaug], dim=0) if imgaug_torch is not None else imgaug

        assert img_torch is not None, "No visualization samples found!"
        self._save_images(img_torch, dump_dir, prefix, global_step, local_rank, vis_uids)
        if imgaug_torch is not None:
            self._save_images(imgaug_torch, dump_dir, f'{prefix}_raw', global_step, local_rank, vis_uids)

        if was_training:
            model.train()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handles visualization during evaluation phase."""
        if self.on_rank_zero_only and not state.is_world_process_zero:
            return

        model = kwargs.pop('model')
        train_dataloader = kwargs.pop('train_dataloader')
        eval_dataloader = kwargs.pop('eval_dataloader')
        dump_dir = os.path.join(args.output_dir, self.dump_prefix)
        os.makedirs(dump_dir, exist_ok=True)

        self._visualize_loop(model, train_dataloader, dump_dir, 'train', state.global_step, args.local_rank)
        self._visualize_loop(model, eval_dataloader, dump_dir, 'eval', state.global_step, args.local_rank)


class ManualWandbCallback(WandbCallback):
    def __init__(self, final_wandb_args: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _original_wandb_init = self._wandb.init
        def _hacked_wandb_init(**init_kwargs):
            del init_kwargs  # forced to use given wandb init args
            print("USING HACKED WANDB INIT METHOD!")
            return _original_wandb_init(**final_wandb_args)
        self._wandb.init = _hacked_wandb_init
