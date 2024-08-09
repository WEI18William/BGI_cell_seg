"""
2023/09/21 @fxzhao 添加测试用例
"""

import pytest
import numpy as np
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[3]))

from cellbin.dnn.cseg.cell_trace import get_trace, get_trace_v2

class TestGetTrace():

    @pytest.mark.parametrize("size", [5000, 11000, 11001, 12000, 20000])
    def test_get_trace(self, size):
        """
        2023/09/21 @fxzhao 主要比较输入的uint8数组,是否转类型为float对计算结果无影响
        """
        arr = np.random.randint(0, 255, (size, size), dtype=np.uint8)

        assert get_trace(arr) == get_trace_v2(arr)

