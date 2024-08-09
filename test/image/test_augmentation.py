"""
2023/09/21 @fxzhao 添加测试用例
"""

import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[2]))

from cellbin.image.augmentation import f_equalize_adapthist, f_histogram_normalization, f_ij_16_to_8, f_ij_16_to_8_v2

from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity

class TestAugmentation():

    @pytest.mark.parametrize("size", [10, 200, 3000])
    def test_clahe(self, size):
        """
        2023/09/21 @fxzhao 比较skimage和cv2的clahe算法,预期结果差异在一定范围内
        """
        arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)

        res1 = rescale_intensity(f_equalize_adapthist(arr, kernel_size=128), out_range=(0, 1))
        res2 = equalize_adapthist(arr, kernel_size=128)
        assert np.max(np.abs(res1-res2)) < 0.1

    @pytest.mark.parametrize("size", [10, 200, 3000])
    def test_rescale_intensity(self, size):
        """
        2023/09/21 @fxzhao 比较rescale_intensity和numba加速版本的结果,理应没有精度差异
        """
        arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)

        res1 = f_histogram_normalization(arr)
        arr = arr.astype('float32')
        res2 = rescale_intensity(arr, out_range=(0.0, 1.0))
        assert np.count_nonzero(res1 != res2) == 0

    @pytest.mark.parametrize("size", [10, 200, 3000])
    def test_16to8(self, size):
        """
        2023/09/21 @fxzhao 比较f_ij_16_to_8和numba加速版本的结果,理应没有精度差异
        """
        arr = np.random.randint(0, 65535, (size, size), dtype=np.uint16)

        res1 = f_ij_16_to_8(arr)
        res2 = f_ij_16_to_8_v2(arr)
        assert np.count_nonzero(res1 != res2) == 0

    @pytest.mark.parametrize("size", [10, 200, 3000])
    def test_16to8_rgb(self, size):
        """
        2023/10/16 @fxzhao 测试三通道图片输入
        """
        arr = np.random.randint(0, 65535, (size, size, 3), dtype=np.uint16)

        res1 = f_ij_16_to_8(arr)
        res2 = f_ij_16_to_8_v2(arr)
        assert np.count_nonzero(res1 != res2) == 0

