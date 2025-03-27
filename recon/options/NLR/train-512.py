import os
from pathlib import Path

from .. import TrainingOptions


os.environ['PBR_ENABLE_DS'] = '1'

option = TrainingOptions(
    run_group=__package__.split('.')[-1],
    run_case=Path(__file__).stem,
    load_from="<PATH_TO_TRAINED_256_RES_MODEL>",  # TODO: replace
    case_training_args=dict(
        per_device_train_batch_size=1,  # TODO: total batch size of 128 (4 nodes of 8 x A100 80G)
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type='cosine',
        max_steps=30000,
        warmup_ratio=0.05,
        eval_on_start=True,
        torch_compile=True,
    ),
    case_dataset_args=dict(
        data_root='<DATA_ROOT_CONTAINING_OBJECT_RENDERINGS>',  # TODO: replace
        meta_file='<META_FILE_WITH_TRAIN_EVAL_SPLITS>',  # TODO: replace with local path to `metadata.json`
        input_resolution=512,
        reference_resolution=256,
        pbr_resolution=512,
        ref_mode='hybrid+0.50',  # accommodate for diffusion aug, set 40% for diffusion and 10% for order
        n_refer_lighting_fixed=3,
        n_refer_lighting_random=6,
        aug_prob=0.6,
        aug_orient_level=(0.1, 0.02),
        aug_intense_level_in=0.1,
        aug_intense_level_ref=(0.9, 1.3, 0.05),
        aug_blur_rate=0.5,
        aug_grid_distortion=(0.15, 0.3),
        aug_diffusion=0.8,  # overall prob of 40% * 55/80 ~ 27.5%, front and 2 random views, 3 out of 5 views
        diff_gen_root='<ROOT_TO_GENERATED_REFERENCE_IMAGES>',  # TODO: replace
    ),
    case_model_args=dict(
        block_out_channels=(224, 448, 672, 896),
        loss_weights=(1.0, 1.0, 1.0, 0.8, 0.2),
    ),
    case_wandb_args=dict(),
    case_visualize_pbr_args=dict(),
    case_deepspeed_args=dict(),
)
