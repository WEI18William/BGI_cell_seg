"""
2023/09/21 @fxzhao 添加测试用例
"""

import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[2]))

from cellbin.contrib.alignment import multiply_sum, AlignByTrack
from cellbin.image.augmentation import f_padding


class TestMultiplySum:
    """
    Test multiply_sum function
    """

    @pytest.mark.parametrize("size", [5, 10, 200, 3000])
    def test_multiply_sum(self, size):
        arr1 = np.random.randint(0, 255, (size, size), dtype=np.uint32)
        arr2 = np.random.randint(0, 255, (size, size), dtype=np.uint32)
        assert np.sum(np.multiply(arr1, arr2).astype(np.uint64)) == multiply_sum(arr1, arr2)

class TestAlignByTrack:

    def old_method(self, transformed_image, vision_image, offset):
        if offset[0] < 0:
            left_x = int(round(abs(offset[0])))
            vision_image = f_padding(vision_image, 0, 0, left_x, 0)
        else:
            vision_image = vision_image[:, int(round(offset[0])):]

        if offset[1] < 0:
            up_y = int(round(abs(offset[1])))
            vision_image = f_padding(vision_image, up_y, 0, 0, 0)
        else:
            vision_image = vision_image[int(round(offset[1])):, :]

        shape_vision = np.shape(vision_image)
        shape_transform = np.shape(transformed_image)

        if shape_vision[0] > shape_transform[0]:
            vision_image = vision_image[:shape_transform[0], :]
        else:
            vision_image = f_padding(vision_image, 0, shape_transform[0] - shape_vision[0], 0, 0)

        if shape_vision[1] > shape_transform[1]:
            vision_image = vision_image[:, :shape_transform[1]]
        else:
            vision_image = f_padding(vision_image, 0, 0, 0, shape_transform[1] - shape_vision[1])
        score = np.sum(np.multiply(vision_image, transformed_image))
        return score
    
    @pytest.mark.parametrize("size", [5, 10, 20, 200])
    def test_cal_score(self, size):
        """
        2023/09/21 @fxzhao 注意:原版本的旧方法有问题,如果数据类型是uint8,那么np.multiply会出现溢出的情况,所以这里使用类型np.uint32
        """
        arr1 = np.random.randint(0, 255, (300, 400), dtype=np.uint32)
        arr2 = np.random.randint(0, 255, (350, 500), dtype=np.uint32)
        offset = [size, -size]
        assert AlignByTrack.cal_score(arr1, arr2, offset) == self.old_method(arr1, arr2, offset)
