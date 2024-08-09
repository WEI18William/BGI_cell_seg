from cellbin.image.augmentation import f_rgb2gray, f_ij_16_to_8_v2, f_ij_auto_contrast_v3
from cellbin.image.augmentation import f_rgb2gray, f_ij_16_to_8_v2
from cellbin.image.mask import f_instance2semantics
from cellbin.image.augmentation import f_percentile_threshold, f_histogram_normalization, f_equalize_adapthist
from cellbin.image.augmentation import f_padding as f_pad
from cellbin.image.morphology import f_deep_watershed
from cellbin.modules import StainType
from cellbin.image.morphology import f_watershed
from cellbin.utils import clog
import pdb

import numpy as np
import cv2
from skimage.exposure import rescale_intensity
import tifffile
from skimage.measure import label, regionprops


def f_prepocess(img, img_type):

    """
    2023/09/20 @fxzhao 优化图像16转8;增加对img的dtype判断,降低类型转换开销;支持传入图片路径
    """
    if isinstance(img, str):
        img = tifffile.imread(img)
    img = np.squeeze(img)
    if img_type.upper() == StainType.rna.value:
        # rna
        img = f_ij_auto_contrast_v3(img)
        img = f_ij_16_to_8_v2(img)
    else:
        # 转录组
        if img.dtype != 'uint8':
            img = f_ij_16_to_8_v2(img)
        # if img.ndim == 3:
        #     img = f_rgb2gray(img, False)
        # if img_type == StainType.HE.value:
        #     img = cv2.bitwise_not(img)
        img = f_percentile_threshold(img)
        #img = f_equalize_adapthist(img, 128)
        img = f_histogram_normalization(img)

    if img.dtype != np.float32:
        img = np.array(img).astype(np.float32)
    img = np.ascontiguousarray(img)
    return img


def f_postpocess(pred):
    pred = pred[0, :, :, 0]

    # pred[pred > 0] = 1
    # pred = np.uint8(pred)

    pred = f_instance2semantics(pred)
    return pred


def f_preformat(img):
    if img.ndim < 3:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def f_preformat_rna(img):
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def f_postformat_rna(pred):
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.squeeze(pred)
    pred = sigmoid(pred)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = np.array(pred, dtype="uint8")
    return pred


def normalize_to_0_255(arr):
    v_max = np.max(arr)
    v_min = np.min(arr)
    if v_max == 0:
        return arr

    # 判断是否存在值在0-255的区间内
    if 0 <= v_min <= 255 or 0 <= v_max <= 255 or (v_max > 255 and v_min < 0):
        # 如果存在，将这些值乘以一个因子
        factor = 1000
        np.multiply(arr, factor)

    # 进行归一化
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return ((arr - arr_min) * 255) / (arr_max - arr_min)


def f_postformat(pred):
    if not isinstance(pred, list):
        pred = [np.uint8(rescale_intensity(pred, out_range=(0, 255)))]
    else:
        pred = [np.uint8(rescale_intensity(pred[0], out_range=(0, 255)))]
    # p_max = np.max(pred[-1])
    pred = pred[0][0, :, :, 0]
    # pred = f_deep_watershed(pred,
    #                         maxima_threshold=round(0.1 * 255),
    #                         maxima_smooth=0,
    #                         interior_threshold=round(0.2 * 255),
    #                         interior_smooth=0,
    #                         fill_holes_threshold=15,
    #                         small_objects_threshold=0,
    #                         radius=2,
    #                         watershed_line=0)
    return pred


def f_postprocess_v2(pred):
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.expand_dims(pred, axis=(0, -1))
    # pred = np.uint64(np.multiply(np.around(pred, decimals=2), 100))
    # pred = np.uint8(normalize_to_0_255(pred))

    pred = f_deep_watershed([pred],
                            maxima_threshold=round(0.1 * 255),
                            maxima_smooth=0,
                            interior_threshold=round(0.2 * 255),
                            interior_smooth=0,
                            fill_holes_threshold=15,
                            small_objects_threshold=0,
                            radius=2,
                            watershed_line=0,
                            maxima_algorithm='h_maxima')
    pred = f_postpocess(pred)
    return pred


def f_preformat_mesmer(img):
    img = np.stack((img, img), axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


def f_check_shape(ct):
    farthest = 0
    max_dist = 0
    for i in range(ct.shape[0]):
        d = np.sqrt((ct[i][0][0] - ct[i - 1][0][0]) ** 2 + (ct[i][0][1] - ct[i - 1][0][1]) ** 2)
        if d > max_dist:
            max_dist = d
            farthest = i
    rect = cv2.minAreaRect(ct)
    if rect[1][0] * rect[1][1] == 0:
        return True
    if rect[1][0] / rect[1][1] >= 3 or rect[1][0] * rect[1][1] <= 1 / 3 or max_dist ** 2 > rect[1][0] * rect[1][1]:
        return True
    return False


def f_contour_interpolate(mask_temp, value=255):
    tmp = mask_temp.copy()
    tmp[tmp == value] = 0
    img = mask_temp.copy()
    img[img != value] = 0
    img[img > 0] = 255
    img = np.uint8(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    img = np.uint16(img)
    img[img > 0] = value
    cv2.drawContours(img, [hull], -1, value, thickness=-1)
    img[img + tmp > value] = 0
    return img


def f_is_border(img, i, j):
    s_r = [-1, 0, 1]
    s_c = [-1, 0, 1]
    if i == 0:
        s_r = s_r[1:]
    if i == img.shape[0] - 1:
        s_r = s_r[:-1]
    if j == 0:
        s_c = s_c[1:]
    if j == img.shape[1] - 1:
        s_c = s_c[:-1]
    for r in s_r:
        for c in s_c:
            if img[i + r][j + c] != 0 and img[i + r][j + c] != img[i][j]:
                return 1
    return 0


def f_border_map(img):
    map = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 0:
                continue
            map[i][j] = f_is_border(img, i, j)
    return map


def f_postprocess_rna(mask):
    clog.info(f"Start rna post processing")
    label_mask = label(mask, connectivity=2)
    props = regionprops(label_mask, label_mask)
    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]].copy()
        if obj['filled_area'] < 80:
            # if obj['filled_area'] < 0:
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
        else:
            tmp_mask = label_mask_temp.copy()
            tmp_mask[tmp_mask != obj['label']] = 0
            tmp_mask, tmp_area = f_watershed(tmp_mask)
            tmp_mask = np.uint32(tmp_mask)
            tmp_mask[tmp_mask > 0] = obj['label']
            label_mask_temp[tmp_area > 0] = tmp_mask[tmp_area > 0]
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][tmp_area > 0] = label_mask_temp[tmp_area > 0]
    label_mask[label_mask > 0] = 255
    label_mask = label(label_mask, connectivity=2)
    props = regionprops(label_mask, label_mask)
    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        label_mask_temp = label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        if obj['filled_area'] < 80:
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
            continue
        tmp_mask = label_mask_temp.copy()
        tmp_mask[tmp_mask != obj['label']] = 0
        tmp_mask[tmp_mask > 0] = 255
        tmp_mask = np.uint8(tmp_mask)
        contours, _ = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if 0 not in contours[0] and f_check_shape(contours[0]):
            label_mask_temp[label_mask_temp == obj['label']] = 0
            label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = label_mask_temp
        intp = f_contour_interpolate(label_mask_temp, obj['label'])
        label_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]][intp == obj['label']] = obj['label']
    map = f_border_map(label_mask)
    label_mask[map > 0] = 0
    label_mask[np.where(label_mask > 0)] = 1
    return np.uint8(label_mask)


def f_postformat_mesmer(pred):
    if isinstance(pred, list):
        pred = [pred[0], pred[1][..., 1:2]]
    pred = f_deep_watershed(pred,
                            maxima_threshold=0.075,
                            maxima_smooth=0,
                            interior_threshold=0.2,
                            interior_smooth=2,
                            small_objects_threshold=15,
                            fill_holes_threshold=15,
                            radius=2,
                            watershed_line=0)
    return f_postpocess(pred)


def f_padding(img, shape, mode='constant'):
    h, w = img.shape[:2]
    win_h, win_w = shape[:2]
    img = f_pad(img, 0, abs(win_h - h), 0, abs(win_w - w), mode)
    return img


def f_fusion(img1, img2):
    img1 = cv2.bitwise_or(img1, img2)
    return img1
