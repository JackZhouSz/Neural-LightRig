import os
import argparse
from omegaconf import OmegaConf
from PIL import Image
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download

from utils.vis import replace_bg_preserving_alpha
from utils.vis import concat_images_vertically


def prepare_stage1_mld(cfg_path: str, ckpt_path: str):
    from mld.model import MVDiffusion

    config = OmegaConf.load(cfg_path)
    model = MVDiffusion.load_from_checkpoint(
        ckpt_path,
        strict=True,
        map_location="cpu",
        **config.model.params).cuda()
    model.pipeline.to(model.device)
    model.eval()
    return model


def prepare_stage2_recon(ckpt_path: str):
    from recon.model import PBRReconConfig, PBRUNetModelForReconstruction

    config = PBRReconConfig.from_pretrained(ckpt_path)
    model = PBRUNetModelForReconstruction.from_pretrained(ckpt_path, config=config).cuda()
    model.eval()
    return model


@torch.no_grad()
def inference_impl(
    model1,
    model2,
    img_path: str,
    save_path: str,
    input_res: int = 512,
    guidance_scale: float = 2.0,
    guidance_rescale: float = 0.7,
    num_inference_steps: int = 75,
):
    img_in_rgba = Image.open(img_path).resize((input_res, input_res))
    img_in_rgba_white = replace_bg_preserving_alpha(img_in_rgba, 255)
    img_in_rgba_black = replace_bg_preserving_alpha(img_in_rgba, 0)
    img_ref: Image = model1.pipeline(
        image=img_in_rgba_white.convert("RGB"),
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    img_rm, img_albedo, img_normal = model2.predict(
        input_image=img_in_rgba_black,
        ref_image=img_ref,
    )
    # stitch the results in one large img, [img_in, img_ref, img_rm, img_albedo, img_normal]
    img_out = Image.new("RGBA", (input_res * 5, input_res))
    img_out.paste(img_in_rgba, (0, 0))
    img_out.paste(img_ref.resize((input_res, input_res)), (input_res, 0))
    img_out.paste(img_rm, (input_res * 2, 0))
    img_out.paste(img_albedo, (input_res * 3, 0))
    img_out.paste(img_normal, (input_res * 4, 0))
    imgs = [img_out]

    # Save the concatenated image
    img_save = concat_images_vertically(imgs)
    img_save.save(save_path)


def convert(root_dir: str, target_row: int = 0, working_res: int = 512):

    combined_dir = os.path.join(root_dir, 'outputs')

    str_target_row = f"_{target_row}" if target_row > 0 else ""
    save_albedo_dir = os.path.join(root_dir, f'albedo{str_target_row}')
    save_roughness_dir = os.path.join(root_dir, f'roughness{str_target_row}')
    save_metallic_dir = os.path.join(root_dir, f'metallic{str_target_row}')
    save_normal_dir = os.path.join(root_dir, f'normal{str_target_row}')
    save_mask_dir = os.path.join(root_dir, f'mask{str_target_row}')
    save_ref_dir = os.path.join(root_dir, f'ref{str_target_row}')

    os.makedirs(save_albedo_dir, exist_ok=True)
    os.makedirs(save_roughness_dir, exist_ok=True)
    os.makedirs(save_metallic_dir, exist_ok=True)
    os.makedirs(save_normal_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_ref_dir, exist_ok=True)


    def _convert_impl(combined_path: str, save_albedo_path: str, save_roughness_path: str, save_metallic_path: str, save_normal_path: str, save_mask_path: str, save_ref_path: str):
        img_combine = Image.open(combined_path)  # RGBA
        img_albedo = img_combine.crop((3 * working_res, target_row * working_res, 4 * working_res, (target_row+1) * working_res)).convert("RGB")  # RGB
        img_rm = img_combine.crop((2 * working_res, target_row * working_res, 3 * working_res, (target_row+1) * working_res))  # RGBA
        img_roughness = img_rm.getchannel("G")
        img_metallic = img_rm.getchannel("B")
        img_normal = img_combine.crop((4 * working_res, target_row * working_res, 5 * working_res, (target_row+1) * working_res)).convert("RGB")
        img_mask = img_combine.crop((0 * working_res, target_row * working_res, 1 * working_res, (target_row+1) * working_res)).getchannel("A")
        img_ref = img_combine.crop((1 * working_res, target_row * working_res, 2 * working_res, (target_row+1) * working_res)).convert("RGB")
        # save
        img_albedo.save(save_albedo_path)
        img_roughness.save(save_roughness_path)
        img_metallic.save(save_metallic_path)
        img_normal.save(save_normal_path)
        img_mask.save(save_mask_path)
        img_ref.save(save_ref_path)


    for img_name in tqdm(os.listdir(combined_dir)):
        combined_path = os.path.join(combined_dir, img_name)
        albedo_path = os.path.join(save_albedo_dir, img_name)
        roughness_path = os.path.join(save_roughness_dir, img_name)
        metallic_path = os.path.join(save_metallic_dir, img_name)
        normal_path = os.path.join(save_normal_dir, img_name)
        mask_path = os.path.join(save_mask_dir, img_name)
        ref_path = os.path.join(save_ref_dir, img_name)

        _convert_impl(combined_path, albedo_path, roughness_path, metallic_path, normal_path, mask_path, ref_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Neural-LightRig Inference Script')
    parser.add_argument('--seed', type=int, default=511, help='Random seed')
    parser.add_argument('--img_dir', type=str, nargs='+', required=True, help='Directory or directories containing input images')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--input_res', type=int, default=512, help='Resolution for input images')
    parser.add_argument('--cfg_scale', type=float, default=2.0, help='Guidance scale for stage1')
    parser.add_argument('--cfg_rescale', type=float, default=0.7, help='Guidance rescale for stage1')
    parser.add_argument('--infer_steps', type=int, default=75, help='Number of inference steps for stage1')
    args = parser.parse_args()

    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare checkpoint
    _root_ckpt_path = snapshot_download(
        repo_id="zxhezexin/neural-lightrig-mld-and-recon",
        repo_type="model",
        local_dir="./ckpt",
        local_dir_use_symlinks=False,
    )

    # prepare models
    _model1_ckpt_path = os.path.join(_root_ckpt_path, "mld.pt")
    _model2_ckpt_path = os.path.join(_root_ckpt_path, "recon")
    model1 = prepare_stage1_mld(
        cfg_path="./mld/configs/infer.yaml",
        ckpt_path=_model1_ckpt_path,
    )
    model2 = prepare_stage2_recon(
        ckpt_path=_model2_ckpt_path,
    )

    # Set input and output directories
    img_dirs = args.img_dir
    save_dir = args.save_dir
    input_res = args.input_res
    guidance_scale = args.cfg_scale
    guidance_rescale = args.cfg_rescale
    inference_steps = args.infer_steps

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'outputs'), exist_ok=True)

    # Get list of image files
    img_files = [os.path.join(img_dir, f) for img_dir in img_dirs for f in os.listdir(img_dir) \
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    # Sort files for consistent ordering
    img_files.sort()

    for img_path in tqdm(img_files):
        # Construct save_path
        img_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, 'outputs', f"{os.path.splitext(img_name)[0]}_pred.png")

        # Perform inference and get metrics
        inference_impl(
            model1=model1,
            model2=model2,
            img_path=img_path,
            save_path=save_path,
            input_res=input_res,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            num_inference_steps=inference_steps,
        )

    # convert to separate outputs
    convert(
        root_dir=save_dir,
        working_res=input_res,
    )


if __name__ == '__main__':
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    main()
