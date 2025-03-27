import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .util import instantiate_from_config
from recon.dataset import PBRReconDataset, under_prob
from utils.proxy import no_proxy


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        max_steps=None,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_steps = max_steps

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):
        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        assert self.max_steps is not None, "max_steps should be set before calling train_dataloader."
        sampler = DistributedSampler(self.datasets['train'])
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        sampler = DistributedSampler(self.datasets['validation'])
        return DataLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class MultiLightDiffusionDataset(PBRReconDataset):
    """
    Reuse dataset from reconstruction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_view(self):
        return 0 if under_prob(self.force_front_view_prob) else np.random.randint(0, self.N_CANDIDATE_VIEWS_PER_UID)

    @no_proxy
    def inner_get_item(self, idx):
        uid = self.uids[idx]
        using_mode, using_n_fixed, using_n_random = self.decode_mode_with_prob()
        view = self.sample_view()

        input_image, _ = self.load_input_image(uid, view)  # possible aug in the input
        _, _, reference_images, _ = self.load_reference_images(
            uid, view,
            using_mode=using_mode, using_n_fixed=using_n_fixed, using_n_random=using_n_random,
        )

        return {
            'cond_imgs': input_image[:3],
            'target_imgs': reference_images,
        }
