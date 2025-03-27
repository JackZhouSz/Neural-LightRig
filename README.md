# Neural-LightRig

[![Neural LightRig](https://img.shields.io/badge/Paper-Arxiv-green)](https://arxiv.org/abs/2412.09593)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Page-Neural%20LightRig-red.svg)](https://projects.zxhezexin.com/neural-lightrig)
[![Model](https://img.shields.io/badge/Model-Hugging%20Face-blue.svg)](https://huggingface.co/zxhezexin/neural-lightrig-mld-and-recon)

This repository contains the official code for the paper **Neural LightRig: Unlocking Accurate Object Normal and Material Estimation with Multi-Light Diffusion**.

![Teaser Video](assets/teaser-video.gif)

## Overview

![Method Overview](assets/method_overview.jpg)

## Setup

### Installation

```
git clone https://github.com/ZexinHe/Neural-LightRig.git
cd Neural-LightRig
```

### Environment

We recommend `python>=3.11` and `torch>=2.4.0`. Not all versions are tested, but the code should work with other versions as well.

Please make sure `torch` is ready before installing the following requirements.
```
pip install -r requirements.txt
```

## Inference

Pretrained models are available on [Hugging Face](https://huggingface.co/zxhezexin/neural-lightrig-mld-and-recon). They will be automatically downloaded when running inference for the first time.

### Prepare Images

Images should be background-removed in advance and put into a folder. We provide example RGBA images in `./assets/examples`.

### Run Inference

Run the following command to perform inference on the example images. The results will be saved in `./results/example_outputs`. Please modify the `--img_dir` and `--save_dir` arguments to your own paths.

```
python inference.py --img_dir "./assets/examples" --save_dir "./results/example_outputs"
```

More inference arguments are available, including seed, classifer-free-guidance, inference steps, and more. Please refer to `inference.py` for more details.

## Training

### Prepare Data

Our rendering dataset `LightProp` is on [Hugging Face](https://huggingface.co/datasets/zxhezexin/NLR-LightProp-Objaverse-Renderings). Please follow instructions there on how to download and prepare the dataset.

### Training Multi-Light Diffusion

Update training [configs](mld/configs/train-phase-1.yaml). Run training with the following example command.

```
torchrun --nproc_per_node=1 -m mld.train --base mld/configs/train-phase-1.yaml
```

Then update checkpoint path for the trained model in [phase-2 config](mld/configs/train-phase-2.yaml). Continue training with the following example command.

```
torchrun --nproc_per_node=1 -m mld.train --base mld/configs/train-phase-2.yaml
```

### Training Reconstruction

Update training configs for [256-resolution](recon/options/NLR/train-256.py). You may either train from scratch or finetune on our checkpoint for your own purposes. Run the following example command to start training.

```
torchrun --nproc_per_node=1 -m recon.trainer --group NLR --case train-256
```

Update checkpoint path for the trained 256-resolution model in [512-resolution config](recon/options/NLR/train-512.py). Continue training with the following example command.

```
torchrun --nproc_per_node=1 -m recon.trainer --group NLR --case train-512
```

## Acknowledgement

We thank the authors of the following repositories for their great works!
- [diffusers](https://github.com/huggingface/diffusers)
- [InstantMesh](https://github.com/TencentARC/InstantMesh)

## Citation

If you find this work useful for your research, please consider citing
```
@misc{neural_lightrig,
    title={Neural LightRig: Unlocking Accurate Object Normal and Material Estimation with Multi-Light Diffusion}, 
    author={Zexin He and Tengfei Wang and Xin Huang and Xingang Pan and Ziwei Liu},
    year={2024},
    eprint={2412.09593},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.09593},
}
```
