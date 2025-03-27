import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from megfile import smart_path_join, smart_open, smart_exists
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange
from transformers import DefaultDataCollator

from utils.proxy import no_proxy
from utils.augs import augment_intensity, augment_grid_distortion, augment_blur


def under_prob(prob: float) -> bool:
    if prob == 0:
        return False
    return random.random() < prob


class PBRReconDataset(Dataset):

    N_CANDIDATE_VIEWS_PER_UID: int = 5
    N_CANDIDATE_INPUTS_PER_VIEW: int = 5
    N_CANDIDATE_REFER_LIGHTING_FIXED: int = 9
    N_CANDIDATE_REFER_LIGHTING_RANDOM: int = 9

    def __init__(
        self,
        data_root: str, meta_file: str, meta_split: str = None,
        input_resolution: int = 512, reference_resolution: int = 256, pbr_resolution: int = 512,
        n_refer_lighting_fixed: int = 9, n_refer_lighting_random: int = 0, ref_mode: str = 'ordered',
        aug_prob: float = 0.0,
        aug_orient_level: Union[float, Tuple[float]] = 0.,
        aug_intense_level_in: Union[float, Tuple[float]] = 0.,
        aug_intense_level_ref: Union[float, Tuple[float]] = 0.,
        aug_blur_rate: float = 1.,
        aug_grid_distortion: Union[float, Tuple[float]] = 0.,
        aug_diffusion: float = 0.,
        diff_gen_root: str = None,
        blend_input_bg: Optional[float] = None,
        force_front_view_prob: float = 0.,
    ):
        self.data_root = data_root
        self.uids = self._load_meta(meta_file, meta_split)
        self.is_train = (meta_split == 'train')
        self.input_resolution = input_resolution
        self.reference_resolution = reference_resolution
        self.pbr_resolution = pbr_resolution
        self.n_refer_lighting_fixed = n_refer_lighting_fixed
        self.n_refer_lighting_random = n_refer_lighting_random
        self.ref_mode = ref_mode
        self.aug_prob = aug_prob
        self.aug_orient_level = aug_orient_level if isinstance(aug_orient_level, tuple) else (aug_orient_level, aug_orient_level)
        self.aug_intense_level_in = aug_intense_level_in if isinstance(aug_intense_level_in, tuple) else (1. - aug_intense_level_in, 1. + aug_intense_level_in, 0.)
        self.aug_intense_level_ref = aug_intense_level_ref if isinstance(aug_intense_level_ref, tuple) else (1. - aug_intense_level_ref, 1. + aug_intense_level_ref, 0.)
        self.aug_blur_rate = aug_blur_rate
        self.aug_grid_distortion = aug_grid_distortion if isinstance(aug_grid_distortion, tuple) else (0., aug_grid_distortion)
        self.aug_diffusion = aug_diffusion
        self.diff_gen_root = diff_gen_root
        self.blend_input_bg = blend_input_bg
        self.force_front_view_prob = force_front_view_prob

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            print(f"[DEBUG-DATASET] Error when loading {self.uids[idx]}")
            return self.__getitem__(idx+1)
            # raise e

    def sample_view(self, uid: str, mode: str):
        assert self.force_front_view_prob == 0., "Forcing front view is not allowed in the reconstruction stage!"
        if self.aug_diffusion > 0. and mode == 'ordered' and under_prob(self.aug_diffusion):
            candidate_views = [
                _view for _view in range(self.N_CANDIDATE_VIEWS_PER_UID)
                if smart_exists(smart_path_join(self.diff_gen_root, uid, f'{_view:03d}.png'))
            ]
            if candidate_views:
                view = random.choice(candidate_views)
                mode += '+'  # 'ordered+'
            else:
                view = np.random.randint(0, self.N_CANDIDATE_VIEWS_PER_UID)
        else:
            view = np.random.randint(0, self.N_CANDIDATE_VIEWS_PER_UID)
        return view, mode

    def decode_mode_with_prob(self):
        """
            'hybrid+{float}': enter 'ordered' or 'shuffled' with a probability for shuffled
        """
        mode = self.ref_mode
        _using_n_fixed = self.n_refer_lighting_fixed
        _using_n_random = self.n_refer_lighting_random
        if mode.startswith('hybrid+'):
            if under_prob(float(mode.split('+')[1])):
                mode = 'shuffled'
            else:
                mode = 'ordered'
                _using_n_fixed = _using_n_fixed + _using_n_random
                _using_n_random = 0
        return mode, _using_n_fixed, _using_n_random

    @no_proxy
    def inner_get_item(self, idx):
        uid = self.uids[idx]
        using_mode, using_n_fixed, using_n_random = self.decode_mode_with_prob()
        view, using_mode = self.sample_view(uid=uid, mode=using_mode)

        input_image, input_image_raw = self.load_input_image(uid, view)
        reference_images, reference_lightings, reference_images_raw, reference_lightings_raw = \
            self.load_reference_images(
                uid, view,
                using_mode=using_mode, using_n_fixed=using_n_fixed, using_n_random=using_n_random,
            )
        pbr_image = self.load_pbr_image(uid, view)

        if self.is_train:
            optional_augs = {}
        else:
            optional_augs = {
                'input_image_raw': input_image_raw,
                'reference_images_raw': reference_images_raw,
                'illumination_raw': reference_lightings_raw,
            }

        return {
            'uid': uid,
            'input_image': input_image,
            'reference_images': reference_images,
            'pbr_image': pbr_image,
            'illumination': reference_lightings,
            **optional_augs,
        }

    def load_input_image(self, uid: str, view: int):
        """
        Load an input image from the given uid and view.
        Optionally apply intensity augmentation under the given probability.
        Return both the augmented and raw images.
        """
        input_folder = smart_path_join(self.data_root, uid, 'rgba_input')
        input_image_id = np.random.randint(0, self.N_CANDIDATE_INPUTS_PER_VIEW)
        input_image_path = smart_path_join(input_folder, f'{view:03d}_{input_image_id:04d}.png')
        input_image = self._load_image(input_image_path, self.input_resolution, bg_color=self.blend_input_bg)  # [4, H, W]
        # augmentations
        input_image_raw = input_image
        if self.aug_intense_level_in != (1., 1., 0.) and under_prob(self.aug_prob):
            input_image[:3] = augment_intensity(input_image[None, :3], global_local_strength=self.aug_intense_level_in)[0]
        return input_image, input_image_raw

    def load_reference_images(self, uid: str, view: int, using_mode: str, using_n_fixed: int, using_n_random: int):
        """
        Load reference images from the given uid and view.
        Optionally apply blur, intensity, grid-distortion augmentations under the given probability.
        Return both the augmented and raw images.
        Allowed modes:
            - 'no': no reference images
            - 'ordered': fixed reference images followed by random reference images
            - 'shuffled': randomly select from fixed and random reference images with an additional mixed shuffle
            - 'ordered+': similar to 'ordered' but with diffusion generated reference images
        """
        if using_mode.endswith('+'):
            mode = using_mode[:-1]
            _tag_diffusion_generated = True
        else:
            mode = using_mode
            _tag_diffusion_generated = False
        _using_n_fixed = using_n_fixed
        _using_n_random = using_n_random
        assert mode in ['no', 'ordered', 'shuffled']
        if mode == 'no':
            return None, None, None, None
        if mode == 'ordered':
            assert _using_n_random == 0, "Random reference images are not allowed in ordered mode"
        reference_image_folder = smart_path_join(self.data_root, uid, 'rgba_output')
        fixed_reference_image_ids = np.random.choice(
            np.arange(self.N_CANDIDATE_REFER_LIGHTING_FIXED),
            size=_using_n_fixed, replace=False,
        ) if mode == 'shuffled' else np.arange(_using_n_fixed)
        random_reference_image_ids = np.random.choice(
            np.arange(self.N_CANDIDATE_REFER_LIGHTING_FIXED, self.N_CANDIDATE_REFER_LIGHTING_FIXED+self.N_CANDIDATE_REFER_LIGHTING_RANDOM),
            size=_using_n_random, replace=False,
        )
        reference_image_ids = np.concatenate((fixed_reference_image_ids, random_reference_image_ids))
        if mode == 'shuffled':
            np.random.shuffle(reference_image_ids)
        reference_images = []
        for reference_image_id in reference_image_ids:
            reference_image_path = smart_path_join(reference_image_folder, f'{view:03d}_{reference_image_id:04d}.png')
            reference_image = self._load_image(reference_image_path, self.reference_resolution, bg_color=1.0)
            reference_images.append(reference_image[:3])
        reference_images = torch.stack(reference_images, dim=0)  # [N, 3, H, W]
        reference_lightings, reference_lightings_raw = self.load_reference_lightings(uid, view, reference_image_ids)
        # augmentations
        reference_images_raw = reference_images
        if _tag_diffusion_generated:
            diff_gen_ref = self._load_image(smart_path_join(self.diff_gen_root, uid, f'{view:03d}.png'), allow_rgb=True)  # [3, 3H, 3W]
            diff_gen_ref = rearrange(diff_gen_ref, 'c (x h) (y w) -> (x y) c h w', x=3, y=3)  # [N, 3, H, W], in which N is 9
            assert max(reference_image_ids) < 9, "The index of reference images should be less than 9 under ordered+ mode"
            diff_gen_ref = diff_gen_ref[reference_image_ids]  # indexing matching number of reference images
            reference_images = diff_gen_ref
        else:
            if self.aug_blur_rate < 1. and under_prob(self.aug_prob):
                reference_images = augment_blur(reference_images, rate=self.aug_blur_rate)
            if self.aug_intense_level_ref != (1., 1., 0.) and under_prob(self.aug_prob):
                reference_images = augment_intensity(reference_images, global_local_strength=self.aug_intense_level_ref)
            if max(self.aug_grid_distortion) > 0 and under_prob(self.aug_prob):
                reference_images = augment_grid_distortion(reference_images, min_max=self.aug_grid_distortion)
        assert reference_images.shape[0] == reference_lightings.shape[0], "The number of reference images and lightings should match"
        return reference_images, reference_lightings, reference_images_raw, reference_lightings_raw

    def load_pbr_image(self, uid: str, view: int):
        """
        Load a PBR image from the given uid and view.
        Ordered as [roughness, metallic, albedo, normal].
        """
        uid_folder = smart_path_join(self.data_root, uid)
        pbr_image_rm = self._load_image(smart_path_join(uid_folder, 'roughness_metallic', f'{view:03d}_0001.png'), self.pbr_resolution, bg_color=1.0)[1:3]
        pbr_image_albedo = self._load_image(smart_path_join(uid_folder, 'albedo', f'{view:03d}_0001.png'), self.pbr_resolution, bg_color=1.0)[:3]
        pbr_image_normal = self._load_image(smart_path_join(uid_folder, 'normal', f'{view:03d}_0001.png'), self.pbr_resolution, bg_color=(0.5, 0.5, 1.0))[:3]
        pbr_image = torch.cat([pbr_image_rm, pbr_image_albedo, pbr_image_normal], dim=0)  # [8, H, W]
        return pbr_image

    def load_reference_lightings(self, uid: str, view: int, lighting_ids: list):
        """
        Load reference lightings from the given uid and view.
        Optionally apply orientation augmentation under the given probability.
        Probability in tuple (theta, phi).
        Return both the augmented and raw lightings.
        Guaranteed that theta should be rounded periodically to [0, 2pi) and phi should be clamped to [0, pi/2].
        """
        reference_lighting_info = json.load(smart_open(smart_path_join(self.data_root, uid, 'lighting.json'), 'r'))
        reference_thetas = [reference_lighting_info['thetas'][lighting_id] for lighting_id in lighting_ids]
        reference_phis = [reference_lighting_info['phis'][lighting_id] for lighting_id in lighting_ids]
        reference_lightings = torch.tensor([reference_thetas, reference_phis], dtype=torch.float32).t()  # [N, 2]
        # augmentations
        reference_lightings_raw = reference_lightings
        if sum(self.aug_orient_level) > 0 and under_prob(self.aug_prob):
            reference_lightings = reference_lightings + \
                torch.randn_like(reference_lightings) * torch.tensor(self.aug_orient_level, dtype=reference_lightings.dtype)
            reference_lightings[:, 0] = reference_lightings[:, 0] % (2 * np.pi)
            reference_lightings[:, 1] = torch.clamp(reference_lightings[:, 1], 0., np.pi / 2)
        return reference_lightings, reference_lightings_raw

    def _load_meta(self, meta_file, meta_split: str | None):
        if meta_file.endswith('.json'):
            with smart_open(meta_file, 'r') as f:
                data = json.load(f)
            if meta_split is not None:
                uids = data[meta_split]
            else:
                assert isinstance(data, list), "The json file must contain a list of uids"
                uids = data
        elif meta_file.endswith('.txt'):
            with smart_open(meta_file, 'r') as f:
                uids = [line.strip() for line in f]
        return uids

    def _load_image(self, image_file: str, resolution: int = None, bg_color: Union[float, Tuple[float]] = None, return_rgba: bool = True, allow_rgb: bool = False):
        """
        Load an image and return a tensor of shape [4, H, W] in range (0, 1).
        Optionally apply resize under given resolution.
        Optionally apply background color blending.
        Optionally choose to return RGB or RGBA.
        """
        rgba = Image.open(smart_open(image_file, 'rb'))
        if resolution is not None and rgba.size != (resolution, resolution):
            rgba = rgba.resize((resolution, resolution), Image.BICUBIC)
        rgba = np.array(rgba)
        rgba = torch.from_numpy(rgba).float() / 255.
        rgba = rgba.permute(2, 0, 1)  # [C, H, W]
        if not allow_rgb:
            assert rgba.shape[0] == 4, "The image must be RGBA format for now"
        if bg_color is not None:
            if isinstance(bg_color, float):
                bg_color = (bg_color, bg_color, bg_color)
            rgba[:3] = rgba[:3] * rgba[3] + torch.tensor(bg_color, dtype=rgba.dtype, device=rgba.device)[:, None, None] * (1 - rgba[3])
        if return_rgba:
            return rgba
        else:
            if bg_color is None:
                raise RuntimeWarning("Using return_rgba=False without bg_color is not recommended")
            return rgba[:3]


@dataclass
class PBRReconDataCollator(DefaultDataCollator):
    """
    Inherits from the DefaultDataCollator and adds functionality to handle string types like 'uid'.
    Strings will be collected into lists instead of being converted to tensors.
    """

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        batch = super().__call__(features, return_tensors)
        first = features[0]
        for k, v in first.items():
            if isinstance(v, str):
                # additional handling of strings
                batch[k] = [f[k] for f in features]
            elif v is None:
                batch[k] = None
        return batch
