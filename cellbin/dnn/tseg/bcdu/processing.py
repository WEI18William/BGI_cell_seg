from cellbin.image.augmentation import f_resize, f_rgb2gray, f_ij_16_to_8_v2, f_ij_auto_contrast, f_ij_auto_contrast_v2
from cellbin.image.threshold import f_th_li, f_th_sauvola

from cellbin.image.augmentation import f_equalize_adapthist, f_histogram_normalization
from cellbin.image.morphology import f_fill_holes
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_objects
import numpy as np
import cv2


def f_prepocess(img, img_type="SSDNA", tar_size=(256, 256)):
    img = np.squeeze(img)
    if str.upper(img_type) == "SSDNA":   #ssDNA 做亮度增强
        img = f_ij_auto_contrast_v2(img)
    if str.upper(img_type) == "RNA":
        img[img > 0] = 255
        img = np.array(img).astype(np.uint8)
    else:
        if img.dtype != 'uint8':
            img = f_ij_16_to_8_v2(img)
    img = f_resize(img, tar_size, "BILINEAR")

    if str.upper(img_type) == "RNA":
        img = f_ij_auto_contrast(img)
    elif str.upper(img_type) == "HE":

        img = img[:, :, 1]
        img = np.bitwise_not(img)

        img = f_ij_auto_contrast(img)

    img = f_histogram_normalization(img)

    img = np.array(img).astype(np.float32)
    img = np.ascontiguousarray(img)
    return img


def f_postpocess(pred, img_type="SSDNA"):
    pred = np.uint8(rescale_intensity(pred, out_range=(0, 255)))
    if str.upper(img_type) == "HE":
        pred = f_th_li(pred)
    elif str.upper(img_type) == "RNA":
        pred = f_th_li(pred)
    elif str.upper(img_type) == 'SSDNA':    #ssDNA不做后处理
        for i in range(5):    # 迭代放大以及做均值滤波，平滑生成的mask边缘
            pred = cv2.resize(pred, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            pred = cv2.blur(pred, (5, 5))

        return pred
    else:
        pred[pred < 64] = 0
        pred = f_th_sauvola(pred, win_size=127, k=0.5, r=128.0)
    pred = remove_small_objects(pred, min_size=64, connectivity=2)
    pred = f_fill_holes(pred, size=64, connectivity=2)
    pred = np.uint8(pred)
    return pred


def f_preformat(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def f_postformat(pred, img_type="SSDNA"):
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.squeeze(pred)
    return f_postpocess(pred, img_type)
