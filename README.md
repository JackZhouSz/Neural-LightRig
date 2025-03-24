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

Images should be background-removed in advance and put into a folder. We provide example images in `./assets/examples`.

### Run Inference

Run the following command to perform inference on the example images. The results will be saved in `./results`. Please modify the `--img_dir` and `--save_dir` arguments to your own paths.

```
python inference.py --img_dir "./assets/examples" --save_dir "./results"
```

More inference arguments are available, including seed, classifer-free-guidance, inference steps, and more. Please refer to `inference.py` for more details.

## Training

### Prepare Data

Our rendering dataset `LightProp` is on [Hugging Face](https://huggingface.co/datasets/zxhezexin/NLR-LightProp-Objaverse-Renderings). Please follow instructions there on how to download and prepare the dataset.

### Run Training

On the way :)

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
