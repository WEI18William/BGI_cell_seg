import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random
import numpy as np
import tifffile
from skimage.exposure import rescale_intensity

# 数据增强函数
def augment(image, mask=None):
    if mask is None:
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.2))),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            # iaa.Affine(rotate=(-180, 180),
            #            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #            translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)})
        ], random_order=True)
        image_aug = seq(image=image)
        img = np.squeeze(image_aug)
        return img
    else:
        segmap = SegmentationMapsOnImage(mask, shape=image.shape)
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.2))),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            # iaa.Affine(rotate=(-180, 180),
            #            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #            translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)})
        ], random_order=True)
        image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
        img = image_aug

        mask = segmap_aug.draw()[0][:, :, 0]
        mask = np.clip(mask, 0, 1).astype(np.uint8)

        img = np.squeeze(img)

        return img, mask

class two_dataset(Dataset):
    def __init__(self, img_list, mask_list, transform=None, is_val=False):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        self.is_val = is_val

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        if img.shape != (256, 256, 3):
            return self.__getitem__((idx + 1) % len(self.img_list))
        
        mask = np.clip(mask, 0, 1)

        seed = random.randint(0, 10)
        if not self.is_val and seed < 8:
            # 只在训练集上进行数据增强
            img, mask = augment(img, mask)

        img = img.astype(np.float32)
        img = rescale_intensity(img, out_range=(0.0, 1.0))
        mask[mask > 0] = 1
        mask = mask.astype(np.float32)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return img, mask
    

class three_dataset(Dataset):
    def __init__(self, img_list, mask_list, unlabel_list, transform=None, is_val=False):
        self.img_list = img_list
        self.mask_list = mask_list
        self.unlabel_list = unlabel_list
        self.transform = transform
        self.is_val = is_val

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        
        if idx >= len(self.img_list):
            label_idx = random.randint(0, len(self.img_list) - 1)
        else:
            label_idx = idx
            
        if idx >= len(self.unlabel_list):
            unlabel_idx = random.randint(0, len(self.unlabel_list) - 1)
        else:
            unlabel_idx = idx
        
        img_path = self.img_list[label_idx]
        mask_path = self.mask_list[label_idx]
        unlabel_path = self.unlabel_list[unlabel_idx]
        
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        unlabel = tifffile.imread(unlabel_path)
        
        mask = np.clip(mask, 0, 1)

        seed = random.randint(0, 10)
        if  seed < 8:
            # 只在训练集上进行数据增强
            img, mask = augment(img, mask)
            unlabel = augment(unlabel)

        img = img.astype(np.float32)
        img = rescale_intensity(img, out_range=(0.0, 1.0))
        unlabel = unlabel.astype(np.float32)
        unlabel = rescale_intensity(unlabel, out_range=(0.0, 1.0))


        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            unlabel = self.transform(unlabel)
            
        img = torch.from_numpy(img).permute(2, 0, 1)
        unlabel = torch.from_numpy(unlabel).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return img, mask, unlabel