import os, sys
import argparse
from omegaconf import OmegaConf
import wandb

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger

from .util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="NLR",
        help="experiment name",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="-1",
        help="gpu ids to use",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging data",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))


if __name__ == "__main__":

    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, 'multi-light-diff', exp_name)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")
    seed_everything(opt.seed)
    wandb.require("core")

    # init configs
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    trainer_config = lightning_config.trainer
    
    trainer_config["accelerator"] = "gpu"
    if opt.gpus == "-1":
        trainer_config["devices"] = -1
    else:
        ngpu = len(opt.gpus.strip(",").split(','))
        trainer_config['devices'] = ngpu

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    _use_strict_loading = True
    model = instantiate_from_config(config.model)
    model.strict_loading = _use_strict_loading
    if trainer_config.get('load_from') is not None:
        print(f"Loading model weights from {trainer_config.load_from}")
        model = model.__class__.load_from_checkpoint(trainer_config.load_from, **config.model.params, strict=_use_strict_loading, map_location=model.device)
        trainer_config.pop('load_from')
    model.logdir = logdir

    # trainer and callbacks
    trainer_kwargs = dict()

    # logger
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir, 
            "version": "0",
        }
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 5000,
            "save_top_k": -1,
        }
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "mld.train.SetupCallback",
            "params": {
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_weight_decay": True,
            }
        },
    }
    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    trainer_kwargs['precision'] = 'bf16-mixed'
    trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
    trainer_kwargs['logger'] = [
        WandbLogger(
            project='neural-lightrig-mld',
            entity=None,
            name=exp_name,
            id=exp_name,
            save_dir=logdir,
            save_code=True,
            resume='allow',
            settings=wandb.Settings(code_dir="."),
        ),
    ]

    # trainer
    trainer = Trainer(**trainer_config, **trainer_kwargs, num_nodes=opt.num_nodes)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup("fit")

    # configure hyper parameters
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

    # run training loop
    model.strict_loading = _use_strict_loading
    trainer.fit(model, data, ckpt_path='last')
