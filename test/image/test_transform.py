"""
2023/09/21 @fxzhao 添加测试用例
"""

import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[2]))

from cellbin.image.transform import ImageTransform

class TestImageTransform():

    @pytest.mark.parametrize("rot_type", [0,1,2,3])
    def test_rot(self, rot_type):
        """
        2023/09/21 @fxzhao 测试pyvips库旋转 0/90/180/270 的结果正确
        """
        arr = np.random.randint(0, 255, (3, 5), dtype=np.uint8)

        it = ImageTransform()
        it.set_image(arr)
        res1 = it.rot90(rot_type, ret_dst=True)
        res2 = np.rot90(arr, k = rot_type)
        assert np.count_nonzero(res1 != res2) == 0
