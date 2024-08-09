"""
2023/09/21 @fxzhao 添加测试用例
"""

import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[2]))

from cellbin.modules.registration import Registration

class TestRegistration():

    def old_register_score(self, regist_img, vis_img):
        regist_img[np.where(regist_img > 1)] = 1
        total = np.sum(vis_img)
        roi_mat = vis_img * regist_img
        roi = np.sum(roi_mat)
        return int(roi * 100 / total)
    
    def test_register_score(self):
        """
        2023/09/21 @fxzhao 测试register_score方法新旧版本结果一致
        """
        arr1 = np.random.randint(0, 255, (10, 20), dtype=np.uint8)
        arr2 = np.random.randint(0, 255, (10, 20), dtype=np.uint8)

        res1 = Registration.register_score(arr1, arr2)
        res2 = self.old_register_score(arr1, arr2)

        assert np.count_nonzero(res1 != res2) == 0
