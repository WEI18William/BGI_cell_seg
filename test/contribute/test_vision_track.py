"""
2023/09/21 @fxzhao 添加测试用例
"""

import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[2]))

from cellbin.contrib.vision_track import get_mass

class TestGetMass():

    @pytest.mark.parametrize("size", [5, 10, 200, 3000])
    def test_get_mass(self, size):
        """
        2023/09/21 @fxzhao 主要比较输入的uint8数组,是否转类型为float对计算结果无影响
        """
        arr1 = np.random.randint(0, 255, (size, size), dtype=np.uint8)
        arr2 = arr1.astype(float)

        assert np.count_nonzero(get_mass(arr1) != get_mass(arr2)) == 0

