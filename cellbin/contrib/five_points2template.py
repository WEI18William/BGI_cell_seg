"""
用于五点模板推导法
|    ☀   |
|☀  ☀  ☀|
|   ☀   |
必须以上这五点
Author: CB🐂🐎 - lizepeng 
Create time: 2024/05/10

Update 2024/06/20
    1. 添加了多次标点拟合模式
    2. 校验五点是否偏差较大

"""

import os
import math
import numpy as np

from copy import deepcopy
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression


TEMPLATE = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
            [240, 300, 330, 390, 390, 330, 300, 240, 420]]


class FivePoints2Template:
    """
    五点法推模板 懂的都懂
    """
    def __init__(self, ):
        self.points_list = list()
        self._scale_x = 1
        self._scale_y = 1
        self._rotate = 0

        self._height = None
        self._width = None

        self._template = list()
        self._template_point = None

    def set_image_size(self, height: int, width: int):
        """

        Args:
            height:
            width:

        Returns:

        """
        self._height = height
        self._width = width

    def get_scale_and_rotate(self):
        """

        Returns: scale_x, scale_y, rotate -- float

        """
        return self._scale_x, self._scale_y, self._rotate

    def get_template(self):
        """

        Returns: template -- np.array([n, 4])

        """
        _template = np.array(self._template)
        _template = _template[
            np.where((_template[:, 0] > 0) & (_template[:, 1] > 0) &
                     (_template[:, 0] < self._width) & (_template[:, 1] < self._height))
        ]
        return _template

    def add_points(self, points):
        """
        添加五点组合
        Args:
            points: (list | array) & len(points) == 5

        Returns:

        """
        flag = self.check_points(points)
        if flag:
            _points = self._point_search(points)
            self.points_list.append(_points)
            if len(self.points_list) == 1:
                self._scale_x, self._scale_y, self._rotate, src_pt = self._points_to_info(np.array(_points))
                self._template_point = deepcopy(src_pt)
            elif len(self.points_list) == 2:
                temp = np.array(self._template.copy())
                _, _, _, src_pt = self._points_to_info(np.array(_points))
                _pts = temp[np.where((temp[:, 2] == src_pt[2]) & (temp[:, 3] == src_pt[3]))]
                dist = cdist(_pts[:, :2], np.array([src_pt[:2]]))
                dst_pt = _pts[dist.argmin()]
                self._finetune_info(src_pt, dst_pt)
            else:
                temp = np.array(self._template.copy())
                pair_points_list = list()
                for _p in self.points_list:
                    _, _, _, src_pt = self._points_to_info(np.array(_p))
                    _pts = temp[np.where((temp[:, 2] == src_pt[2]) & (temp[:, 3] == src_pt[3]))]
                    dist = cdist(_pts[:, :2], np.array([src_pt[:2]]))
                    dst_pt = _pts[dist.argmin()]
                    pair_points_list.append([src_pt, dst_pt])
                pair_points = np.array(pair_points_list)
                self._caculate_scale_and_rotate(pair_points[:, 1], pair_points[:, 0])

            self._point_inference(self._template_point, [self._height, self._width])
            print("Add points succeed.")
        else:
            print("Add points failed.")

    def check_points(self,
                     points,
                     rotate_max = 30,
                     scale_minmax = [0.4, 2],
                     included_angle_dif = 1,
                     center_point_dif_max = 5,

                      ):
        """
        检查点是否符合规范
        Args:
            points:
            rotate_max: 角度最大限制 -- 30°
            scale_minmax: 尺度限制 -- [0.4, 2]
            included_angle_dif: 夹角最大限制 -- 1°
            center_point_dif_max: 上下左右点与中心点在同一线段的偏差 -- 5 pixel

        Returns:

        """
        if len(points) == 5:
            _points = self._point_search(points)

            dis1 = self.get_distance_from_point_to_line(_points[0], _points[1], _points[2])
            dis2 = self.get_distance_from_point_to_line(_points[0], _points[3], _points[4])

            if dis1 > center_point_dif_max or dis2 > center_point_dif_max:
                # 中心点距离判断
                return 0

            rotate1 = self.fitting_line2rotate(_points[:3])
            rotate2 = self.fitting_line2rotate(np.concatenate((np.array([_points[0]]), np.array(_points[3:]))))
            if np.abs(rotate1) > rotate_max: return 0  # 角度最大
            if np.abs(rotate1 - rotate2) - 90 > included_angle_dif: return 0  # 夹角最大

            rate_list = self.get_template_rate()
            rate_x = cdist([_points[0]], [_points[2]]) / cdist([_points[0]], [_points[1]])
            rate_y = cdist([_points[0]], [_points[4]]) / cdist([_points[0]], [_points[3]])
            
            if rate_x < min(rate_list) - 0.1 or rate_x > max(rate_list) + 0.1 or \
                    rate_y < min(rate_list) - 0.1 or rate_y > max(rate_list) + 0.1:
                # 尺度比例是否违规
                return 0
            index_x = np.abs(np.array(rate_list) - rate_x).argmin()
            index_y = np.abs(np.array(rate_list) - rate_y).argmin()
            scale_x = cdist([_points[0]], [_points[2]]) / TEMPLATE[0][index_x]
            scale_y = cdist([_points[0]], [_points[4]]) / TEMPLATE[0][index_y]
            
            if scale_x < scale_minmax[0] or scale_x > scale_minmax[1] or \
                scale_y < scale_minmax[0] or scale_y > scale_minmax[1:]:
                # 尺度是否超出界限
                return 0
        else:
            # 点数不对
            print("Insufficient number of points.")
            return 0
        return 1

    def _finetune_info(self, src_pt, dst_pt):
        """
        根据点调整scale和rotate
        Args:
            src_pt:
            dst_pt:

        Returns:

        """
        src_pt = np.array(src_pt)[:2]
        dst_pt = np.array(dst_pt)[:2]

        dst_y = dst_pt[1] - self._template_point[1]
        dst_x = dst_pt[0] - self._template_point[0]

        src_y = src_pt[1] - self._template_point[1]
        src_x = src_pt[0] - self._template_point[0]

        rotation_dst = math.degrees(
            math.atan(dst_y / dst_x))
        rotation_src = math.degrees(
            math.atan(src_y / src_x))
        self._rotate -= rotation_dst - rotation_src

        dst_dis = math.sqrt(dst_y ** 2 + dst_x ** 2)
        dst_new_x = -dst_dis * np.cos(np.radians(rotation_src))
        dst_new_y = -dst_dis * np.sin(np.radians(rotation_src))

        self._scale_x *= np.abs(src_x / dst_new_x)
        self._scale_y *= np.abs(src_y / dst_new_y)

    def _points_to_info(self, points):
        """

        Args:
            points:

        Returns:

        """
        rotate = self.fitting_line2rotate(points[:3])
        rate_list = self.get_template_rate()
        x_dist = [cdist([points[0]], [points[1]]), cdist([points[0]], [points[2]])]
        y_dist = [cdist([points[0]], [points[3]]), cdist([points[0]], [points[4]])]
        rate_x = x_dist[1] / x_dist[0]
        rate_y = y_dist[1] / y_dist[0]
        index_x = np.abs(np.array(rate_list) - rate_x).argmin()
        index_y = np.abs(np.array(rate_list) - rate_y).argmin()
        scale_x = cdist([points[0]], [points[2]]) / TEMPLATE[0][index_x]
        scale_y = cdist([points[0]], [points[4]]) / TEMPLATE[0][index_y]

        return scale_x[0][0], scale_y[0][0], rotate, (points[0][0], points[0][1], index_x, index_y)

    def _point_inference(self, src_pt: tuple, region: tuple):
        """
        search stand template from bin file by key(chip_no).
        src_pt :(x, y, ind_x, ind_y)
        region: (height, width)
        """
        if len(self._template) > 0:
            self._template = list()

        x0, y0, ind_x, ind_y = src_pt

        k0 = np.tan(np.radians(self._rotate))
        if k0 == 0: k0 = 0.00000001
        k1 = -1 / k0

        y_intercept0 = y0 - k0 * x0
        x_intercept0 = (y0 - k1 * x0) * k0

        dy = abs(k0 * region[1])
        y_region = (-dy, region[0] + dy)
        dx = abs(k0 * region[0])
        x_region = (-dx, region[1] + dx)

        self.y_intercept = self._get_intercept(self._scale_y, y_intercept0, y_region, ind_y, TEMPLATE[1])
        self.x_intercept = self._get_intercept(self._scale_x, x_intercept0, x_region, ind_x, TEMPLATE[0])
        self._create_cross_points(k0)

    def _caculate_scale_and_rotate(self, points_re, points_qc, center=None):
        '''
        使用模板点和QC点计算出scale和rotate
        '''
        if points_re.shape[1] == 4:
            points_re = points_re[:, :2]

        if center is None:
            center = self._template_point

        points_re[:, 0] -= center[0]
        points_re[:, 1] = center[1] - points_re[:, 1]

        points_qc[:, 0] -= center[0]
        points_qc[:, 1] = center[1] - points_qc[:, 1]

        if points_qc.shape[0] == 1 and \
                points_qc[0, 0] == 0 and \
                points_qc[0, 1] == 0:
            return

        points_re = np.round(points_re, 2)
        points_qc = np.round(points_qc, 2)

        # 最小值距离优化法
        para = self._leastsq_to_scale_and_rotate(points_re, points_qc)
        if para is None:
            return
        _scale_x, _scale_y, _rotate = para.x

        self._rotate -= np.around(_rotate, 5)
        self._scale_x *= (1 + np.around(_scale_x, 5))
        self._scale_y *= (1 + np.around(_scale_y, 5))

    def _get_intercept(self, scale, intercept0, region, ind, templ):
        idx = intercept0
        intercept = [[idx, ind]]
        s, e = region
        item_count = len(templ)
        # face to large
        while idx < e:
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self._rotate))
            idx += item_len
            intercept.append([idx, (ind + 1) % item_count])
            ind += 1
        # face to small
        idx, ind = intercept[0]
        while idx > s:
            ind -= 1
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self._rotate))
            idx -= item_len
            intercept.append([idx, ind])
        return sorted(intercept, key=(lambda x: x[0]))

    def _create_cross_points(self, k):
        for x_ in self.x_intercept:
            for y_ in self.y_intercept:
                x, ind_x = x_
                y, ind_y = y_
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                self._template.append([x0, y0, ind_x, ind_y])

    @staticmethod
    def get_template_rate():
        """
        模板比例
        Returns:

        """
        rate_list = list()
        temp = TEMPLATE[0]
        for i in range(len(temp)):
            rate_list.append(temp[i] / temp[i - 1])

        return rate_list

    @staticmethod
    def _point_search(points):
        """
        Args:
            points:
        """
        points = np.array(points)
        x_mean = np.mean(points[:, 0])
        y_mean = np.mean(points[:, 1])
        return [sorted(points, key=lambda x: ((x[0] - x_mean) ** 2) + (x[1] - y_mean) ** 2)[0],
                points[points[:, 0].argmin()],
                points[points[:, 0].argmax()],
                points[points[:, 1].argmin()],
                points[points[:, 1].argmax()]]

    @staticmethod
    def fitting_line2rotate(points):
        """
        点拟合线段并获得角度
        """
        points = np.array(points)
        line_model = LinearRegression()
        _x = points[:, 0].reshape(-1, 1)
        _y = points[:, 1].reshape(-1, 1)

        line_model.fit(_x, _y)
        _k = line_model.coef_[0][0]
        _rotate = np.degrees(np.arctan(_k))

        return _rotate

    @staticmethod
    def _leastsq_to_scale_and_rotate(point_re, point_qc, method="nelder-mead"):
        '''最小化模板点和QC点距离 并求解出结果'''
        # point_re = np.array([[61.237, 35.355], [-35.355, 61.237], [-61.237, -35.355], [35.355, -61.237]])
        # point_qc = np.array([[100, 100], [-100, 100], [-100, -100], [100, -100]])
        from scipy.optimize import leastsq, minimize

        for k, point in enumerate(point_re):
            if point[0] == 0 and point[1] == 0:
                point_re = np.delete(point_re, k, axis=0)
                point_qc = np.delete(point_qc, k, axis=0)
                break

        if len(point_re) == 0 or len(point_qc) == 0:
            return None

        def _error(p, point_re, point_qc):
            _scale_x, _scale_y, _rotate = p

            src_x = point_re[:, 0]
            src_y = point_re[:, 1]

            _t = (src_y) / (src_x + 0.000000000000001)
            _d = [math.atan(i) for i in _t]
            rotation_src = np.array([math.degrees(i) for i in _d])

            src_x = point_re[:, 0] * (1 + _scale_x)
            src_y = point_re[:, 1] * (1 + _scale_y)

            dis = [math.sqrt(i) for i in src_x ** 2 + src_y ** 2]

            dst_x = dis * np.array([math.cos(math.radians(np.abs(i))) for i in (rotation_src + _rotate)])
            dst_y = dis * np.array([math.sin(math.radians(np.abs(i))) for i in (rotation_src + _rotate)])

            dst_x = [-i if point_re[k, 0] < 0 else i for k, i in
                     enumerate(dst_x)]
            dst_y = [-i if point_re[k, 1] < 0 else i for k, i in
                     enumerate(dst_y)]

            error = (point_qc[:, 0] - dst_x) ** 2 + (point_qc[:, 1] - dst_y) ** 2
            error = [math.sqrt(i) for i in error]
            # print(np.sum(error))
            return np.sum(error) * 0 + max(error) * 1

        para = minimize(_error, x0=np.zeros(3, ), args=(point_re, point_qc), method=method,
                        options={'maxiter': 100})

        return para

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        """
        计算点到直线的距离
        Args:
            point:
            line_point1:
            line_point2:
        """
        # 对于两点坐标为同一点时,返回点与点的距离
        if isinstance(line_point1, np.ndarray):
            flag = (line_point1 == line_point2).all()
        else:
            flag = line_point1 == line_point2
        if flag:
            point_array = np.array(point)
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array - point1_array)
        # 计算直线的三个参数
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        # 根据点到直线的距离公式计算距离
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
        return distance


if __name__ == "__main__":
    pt = FivePoints2Template()
    points_1 = [[3238, 1934], [2947, 2229], [3243, 2460],
                [3474, 2224], [3241, 2226]]
    # points_2 = [[26080.147058823528, 27625.735294117647], [26076.923076923074, 28052.747252747253],
    #             [25649.738219895287, 28049.214659685862], [26072.251308900522, 28520.942408376963],
    #             [26544.55958549223, 28058.031088082902]]
    pt.set_image_size(42080, 41546)
    pt.add_points(points_1)
    # pt.add_points(points_2)
    template = pt.get_template()
    scale_x, scale_y, rotate = pt.get_scale_and_rotate()
    print(1)
