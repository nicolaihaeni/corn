from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2no_norm_im(input_image, imtype=np.uint8):
    sflag = False
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    #print("image shape:",image_numpy.shape)
    if image_numpy.ndim == 2:
        sflag = True
        #image_numpy = np.tile(image_numpy, (3, 1, 1))
    #print("after tile:",image_numpy.shape)
    return image_numpy.astype(imtype)


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    sflag = False
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    #print("image shape:",image_numpy.shape)
    if image_numpy.ndim == 2:
        sflag = True
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    #print("after tile:",image_numpy.shape)
    if image_numpy.shape[0] == 1:
        # Mask image
        image_numpy = np.round(np.transpose(image_numpy, (1, 2, 0)) * 255.0)
    else:
        # RGB image
        image_numpy = np.round((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0)
    if sflag:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def segment_image(rgb_image, background_color='white'):
    if background_color == 'white':
        if rgb_image.shape[-1] == 3:
            rgb_image = rgb_image.transpose(2, 0, 1) / 255.
        # Compute the segmentation of an rgb image
        # return 1 - (rgb_image.sum(0) >= 2.997).float().unsqueeze(0)
        return 1 - (rgb_image.sum(0) >= 2.997).astype(np.float)
    else:
        if rgb_image.shape[-1] == 3:
            rgb_image = rgb_image.transpose(2, 0, 1) / 255.
        return 1 - (rgb_image.sum(0) <= 0.003).astype(np.float)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    if 'cls.png' in image_path:
        rgb = []
        for ucolor in ucolors:
            rgb.append([int(ucolor[i:i+2], 16) for i in (0, 2, 4)])
        h,w = image_numpy.shape[0:2]
        nim = np.zeros((h,w,3),np.uint8)
        unique = np.unique(image_numpy)
        unique = sorted(unique)
        #print("all colors:",unique)
        for ii, num in enumerate(unique):
            mask = image_numpy == num
            nim[mask] = rgb[num]
        cimage_path = image_path.replace('.png','_backup.png')
        nim_pil = Image.fromarray(nim)
        nim_pil.save(image_path)
        image_pil = Image.fromarray(np.squeeze(image_numpy))
        image_pil.save(cimage_path)
    else:
        image_pil = Image.fromarray(np.squeeze(image_numpy))
        image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
