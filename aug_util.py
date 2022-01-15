import torch
import torch.utils.data
import torchvision
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kornia.morphology
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def pil2cv(pil_img: Image):
    cv_img = np.array(pil_img)
    cv_img = cv_img[:, :, ::-1].copy()
    return cv_img


def cv2pil(cv_img: np.ndarray):
    return Image.fromarray(cv_img)


class Resize_Pad(object):
    def __init__(self, size: int):
        self.size = size

    def __call__(self, pil_img: Image):
        im = pil2cv(pil_img)

        old_size = im.shape[:2]  # old_size is in (height, width) format

        ratio = float(self.size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = self.size - new_size[1]
        delta_h = self.size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        new_im = cv2pil(new_im)
        return new_im


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.transforms = [
            kornia.morphology.dilation,
            kornia.morphology.erosion,
            kornia.morphology.closing,
            kornia.morphology.opening]

        self.kernel = torch.randn(5, 5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.kernel = self.kernel.to(x.device)
        t = random.choice(self.transforms)
        x = t(x, self.kernel)
        return x


class GaussianBlur:
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def check_aug_data(dataloader):
    for images, labels in tqdm(dataloader):
        show(torchvision.utils.make_grid(images))
        plt.show()
