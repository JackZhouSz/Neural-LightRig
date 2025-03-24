import os
from importlib import import_module
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import TrainingArguments
import wandb


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint-')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    if len(checkpoints) > 0:
        return os.path.join(output_dir, checkpoints[-1])
    return None


@dataclass(frozen=True)
class TrainingOptions:

    run_group: str = field()
    run_case: str = field()
    project_name: str = field(default='pbr-recon')
    load_from: Optional[str] = field(default=None)
    case_training_args: Dict[str, Any] = field(default_factory=dict)
    case_dataset_args: Dict[str, Any] = field(default_factory=dict)
    case_model_args: Dict[str, Any] = field(default_factory=dict)
    case_wandb_args: Dict[str, Any] = field(default_factory=dict)
    case_visualize_pbr_args: Dict[str, Any] = field(default_factory=dict)
    case_deepspeed_args: Dict[str, Any] = field(default_factory=dict)

    @cached_property
    def run_id(self) -> str:
        return f"{self.run_group}-{self.run_case}"

    @cached_property
    def training_args(self) -> TrainingArguments:
        _ds_cfg = self.deepspeed_args if os.getenv('PBR_ENABLE_DS', False) else None
        _default_args = dict(
            output_dir=f'./results/{self.project_name}/{self.run_group}/{self.run_case}',
            logging_dir=f"./logs/{self.project_name}/{self.run_group}/{self.run_case}",
            run_name=self.run_id,
            max_steps=30000,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            learning_rate=5e-5,
            weight_decay=0.01,
            adam_beta2=0.999,
            lr_scheduler_type='constant',
            bf16=True,
            ddp_backend='nccl',
            batch_eval_metrics=False,
            prediction_loss_only=False,
            eval_strategy='steps',
            eval_on_start=True,
            eval_steps=500,
            logging_steps=50,
            save_steps=500,
            save_total_limit=4,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_num_workers=16,
            dataloader_prefetch_factor=4,
            dataloader_persistent_workers=True,
            deepspeed=_ds_cfg,
        )
        merged_training_args = {**_default_args, **self.case_training_args}
        latest_ckpt = find_latest_checkpoint(merged_training_args['output_dir'])
        if latest_ckpt is not None:
            merged_training_args.update(eval_on_start=False)
        merged_training_args.update(resume_from_checkpoint=latest_ckpt)
        return TrainingArguments(**merged_training_args)

    @cached_property
    def deepspeed_args(self) -> Dict[str, Any]:
        _default_args = dict(
            zero_optimization={
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            gradient_accumulation_steps="auto",
            gradient_clipping="auto",
            train_batch_size="auto",
            train_micro_batch_size_per_gpu="auto",
            bf16={
                "enabled": "auto",
            },
        )
        return {**_default_args, **self.case_deepspeed_args}

    @cached_property
    def model_args(self) -> Dict[str, Any]:
        _default_args = dict(
            in_channels=4+9*3,
            out_channels=1+1+3+3,
            center_input_sample=True,
            num_illumination_params=9*2,
            down_block_types = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            block_out_channels = (224, 448, 672, 896),
            resnet_time_scale_shift = "scale_shift",
        )
        return {**_default_args, **self.case_model_args}

    @cached_property
    def dataset_args(self) -> Dict[str, Any]:
        _default_args = dict(
            input_resolution=512,
            reference_resolution=256,
            pbr_resolution=512,
            ref_mode='ordered',
        )
        return {**_default_args, **self.case_dataset_args}

    @cached_property
    def wandb_args(self) -> Dict[str, Any]:
        _default_args = dict(
            entity='neural-lightrig-recon',
            project=self.project_name,
            name=self.run_case,
            group=self.run_group,
            id=self.run_id,
            save_code=True,
            resume='allow',
            settings=wandb.Settings(code_dir="."),
        )
        return {**_default_args, **self.case_wandb_args}

    @cached_property
    def visualize_pbr_args(self) -> Dict[str, Any]:
        _default_args = dict(
            n_visualize_samples=16,
        )
        return {**_default_args, **self.case_visualize_pbr_args}

    def __post_init__(self):
        assert self.run_group is not None
        assert self.run_case is not None
        assert self.project_name is not None
        assert self.case_training_args is not None
        assert self.case_dataset_args is not None
        assert self.case_model_args is not None
        assert self.case_wandb_args is not None
        assert self.case_visualize_pbr_args is not None
        assert self.case_deepspeed_args is not None
        assert (self.dataset_args.get('ref_mode') == 'no') == (self.model_args.get('num_illumination_params') is None), \
            "num_illumination_params should be None when ref_mode is 'no'"


def load_training_options(group: str, case: str) -> TrainingOptions:
    # dynamic import
    case_option = import_module(f".{group}.{case}", package=__package__).option
    assert isinstance(case_option, TrainingOptions)
    return case_option
