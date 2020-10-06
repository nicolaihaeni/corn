import argparse
import h5py
import os
import glob
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trafo
import numpy as np
import imageio
import lpips
from pytorch_ssim import ssim


def main():
    args = get_args()
    img_size = args.eval_image_size
    src_dir = args.src_dir
    num_samples = args.num_samples

    l1_loss_fn = nn.L1Loss()
    loss_fn_vgg = lpips.LPIPS(net='vgg')

    l1_results = np.empty(num_samples)
    ssim_results = np.empty(num_samples)
    lpips_results = np.empty(num_samples)

    transforms = trafo.Compose([trafo.ToTensor(),
                                trafo.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    for idx in range(num_samples):
        idx_str = "{:05d}".format(idx)

        # reads images as (height,width,channel)
        im_pred = imageio.imread("{}/{}_pred.png".format(src_dir, idx_str))
        im_true = imageio.imread("{}/{}_tgt.png".format(src_dir, idx_str))

        # normalize the values to be [-1,1]
        im_pred_torch = transforms(im_pred).unsqueeze(0)
        im_true_torch = transforms(im_true).unsqueeze(0)

        # resize images to be at the desired resolution if not already
        if not args.no_resize and (im_pred_torch.shape[-1] != img_size or im_true_torch.shape[-1] != img_size):
            im_pred_torch = F.interpolate(im_pred_torch, (img_size, img_size))
            im_true_torch = F.interpolate(im_true_torch, (img_size, img_size))

        # compute l1 error
        l1_loss = l1_loss_fn(im_pred_torch, im_true_torch)
        l1_results[idx] = l1_loss

        # compute ssim error
        ssim_loss = ssim(im_pred_torch, im_true_torch)
        ssim_results[idx] = ssim_loss

        # Compute lpips score
        lp = loss_fn_vgg(im_pred_torch, im_true_torch)
        lpips_results[idx] = lp

        if idx % 1000 == 0:
            print(f"{idx}, im dim: {im_pred_torch.shape[-1]} -- l1, ssim loss, lpips: {l1_loss, ssim_loss, lp}")


    print(f"L1 loss mean: {l1_results.mean()}, std: {l1_results.std()}")
    print(f"SSIM loss mean: {ssim_results.mean()}, std: {ssim_results.std()}")
    print(f"LPIPS score: {lpips_results.mean()}, std: {lpips_results.std()}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='/home/')
    parser.add_argument('--num_samples', type=int, default=20000)
    parser.add_argument('--eval_image_size', type=int, default=128)
    parser.add_argument('--no_resize', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    main()
