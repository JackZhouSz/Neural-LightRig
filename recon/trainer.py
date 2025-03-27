import os
import argparse
from PIL import Image
import numpy as np
import torch
import wandb
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

from .model import pbr_recon_model_init_factory
from .dataset import PBRReconDataset, PBRReconDataCollator
from .options import load_training_options
from .callbacks import VisualizePBRReconCallback, ManualWandbCallback


def compute_metrics(eval_pred: EvalPrediction):
    predictions, _ = eval_pred
    losses = {k: v.mean().item() for k, v in predictions.items()}
    return losses


class PBRReconTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='default_group')
    parser.add_argument('--case', type=str, default='default_case')
    args = parser.parse_args()
    options = load_training_options(args.group, args.case)
    del args

    wandb.require("core")

    training_callbacks = [
        VisualizePBRReconCallback(**options.visualize_pbr_args),
        ManualWandbCallback(final_wandb_args=options.wandb_args),
    ]
    train_dataset = PBRReconDataset(meta_split='train', **options.dataset_args)
    eval_dataset = PBRReconDataset(meta_split='eval', **options.dataset_args)
    trainer = PBRReconTrainer(
        args=options.training_args,
        model_init=pbr_recon_model_init_factory(load_from=options.load_from, **options.model_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PBRReconDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=training_callbacks,
    )
    trainer.train(resume_from_checkpoint=options.training_args.resume_from_checkpoint)
