import numpy as np
from copy import deepcopy
from numba import njit, prange

from cellbin.utils import clog
from cellbin.modules import CellBinElement
from cellbin.contrib.alignment import AlignByTrack
from cellbin.contrib.alignment2he import AlignByTrack as AlignByTrack2HE

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from typing import Any, Dict, Union, Tuple


class Registration(CellBinElement):
    def __init__(self):
        super(Registration, self).__init__()
        self.align_track = AlignByTrack()

        self.offset = [0, 0]
        self.rot90 = 0
        self.flip = False
        self.score = 0

        self.regist_img = np.array([])
        self.fov_transformed = np.array([])
        self.dist_shape = ()
        self.vision_cp = np.array([])
        self.adjusted_stitch_template = np.array([])
        self.adjusted_stitch_template_unflip = np.array([])
        # self._moving_image = self._moving_marker = None
        # self._fixed_image = self._fixed_marker = None
        # self.x_scale = self.y_scale = None
        # self.rotation = None

    # def fixed_image_shape(self, ):
    #     return self._fixed_image.shape
    def set_stain_type(self, stain_type):
        if stain_type == 'HE':
            self.align_track = AlignByTrack2HE()
        else:
            self.align_track = AlignByTrack()
        clog.info(f"Init regis (tissue-{stain_type}) finished.")

    def mass_registration_trans(
            self,
            fov_transformed,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            fov_stitched_shape,
            rotation,
            flip

    ):
        """
        This func is used to regist the transform image

        Args:
            fov_transformed (): transform image
            vision_image (): gene matrix image
            chip_template (): chip template, SS2, FP1
            track_template (): stitch template
            scale_x (): x direction scale
            scale_y (): y direction scale
            fov_stitched_shape (): stitched image shape
            rotation (): rotation
            flip (): if flip

        Returns:
            self.flip: if flip
            self.rot90: 90 degree rot times
            self.offset: offset in x, y direction
            self.score: regist score

        """
        self.align_track.set_chip_template(chip_template=chip_template)
        self.fov_transformed = fov_transformed
        self.dist_shape = vision_image.shape
        track_template_copy = deepcopy(track_template)
        self.adjusted_stitch_template = self.align_track.adjust_cross(
            stitch_template=track_template,
            scale_x=scale_x,
            scale_y=scale_y,
            fov_stitched_shape=fov_stitched_shape,
            new_shape=self.fov_transformed.shape,
            chip_template=chip_template,
            rotation=rotation
        )
        self.adjusted_stitch_template_unflip = self.align_track.adjust_cross(
            stitch_template=track_template_copy,
            scale_x=scale_x,
            scale_y=scale_y,
            fov_stitched_shape=fov_stitched_shape,
            new_shape=self.fov_transformed.shape,
            chip_template=chip_template,
            rotation=rotation,
            flip=False
        )
        self.vision_cp = self.align_track.find_track_on_vision_image(vision_image, chip_template)

        offset, rot_type, score = self.align_track.run(
            transformed_image=self.fov_transformed,
            vision_img=vision_image,
            vision_cp=self.vision_cp,
            stitch_tc=self.adjusted_stitch_template,
            flip=flip,
        )

        # result update
        self.flip = flip
        self.rot90 = rot_type
        self.offset = offset
        self.score = score

        return 0

    def mass_registration_stitch(
            self,
            fov_stitched,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            rotation,
            flip

    ):
        """
        This func is used to regist the stitch image

        Args:
            fov_stitched (): stitched image
            vision_image (): gene matrix image
            chip_template (): chip template, SS2, FP1
            track_template (): stitch template
            scale_x (): x direction scale
            scale_y (): y direction scale
            rotation (): rotation
            flip (): if flip


        Returns:
            self.flip: if flip
            self.rot90: 90 degree rot times
            self.offset: offset in x, y direction
            self.score: regist score

        """
        # transform
        self.fov_transformed = self.stitch_to_transform(
            fov_stitch=fov_stitched,
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=rotation
        )
        fov_stitched_shape = fov_stitched.shape
        self.mass_registration_trans(
            self.fov_transformed,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            fov_stitched_shape,
            rotation,
            flip
        )

        return 0

    @staticmethod
    def stitch_to_transform(fov_stitch, scale_x, scale_y, rotation):
        """
        From stitched image to transform image based on provided scale and rotation

        Args:
            fov_stitch (): stitched image
            scale_x (): x direction scale
            scale_y (): y direction scale
            rotation (): rotation

        Returns:
            fov_transformed: tranformed image

        """
        from cellbin.image.transform import ImageTransform
        i_trans = ImageTransform()
        i_trans.set_image(fov_stitch)
        fov_transformed = i_trans.rot_scale(
            x_scale=scale_x,
            y_scale=scale_y,
            angle=rotation
        )
        return fov_transformed

    def transform_to_regist(self, ):
        """
        From transform image to regist image based on regist result

        Returns:
            self.regist_img: regist image

        """
        from cellbin.image.transform import ImageTransform
        i_trans = ImageTransform()
        i_trans.set_image(self.fov_transformed)
        if self.flip:
            i_trans.flip(
                flip_type='hor',
                ret_dst=False
            )
        i_trans.rot90(self.rot90, ret_dst=False)
        self.regist_img = i_trans.offset(self.offset[0], self.offset[1], self.dist_shape)

    @staticmethod
    def register_score(regist_img, vis_img):
        """
        Calculate regist score baed on gene matrix image

        Args:
            regist_img (): regist image
            vis_img (): gene matrix image

        Returns:
            regist score

        2023/09/20 @fxzhao 使用numba加速计算,并降低内存,不会再修改输入的regist_img
        """
        # regist_img[np.where(regist_img > 1)] = 1
        # total = np.sum(vis_img)
        # roi_mat = vis_img * regist_img
        # roi = np.sum(roi_mat)
        total = np.sum(vis_img)
        roi = multiply_sum(regist_img, vis_img)
        return int(roi * 100 / total)


class RegistrationByNanChip:
    """
    空芯片配准
    """
    def __init__(self):
        pass

    @staticmethod
    def get_lt_zero_point(template_points, x_index=0, y_index=0):
        """
        Args:
            template_points: np.array, 模板点 -- shape == (*, 4)
            x_index:
            y_index:
        Returns:
            zero_template_points: np.array
        """
        zero_template_points = template_points[(template_points[:, 3] == y_index) &
                                               (template_points[:, 2] == x_index)][:, :2]
        return zero_template_points

    @staticmethod
    def get_register_param(chip_lt_point, image_lt_point, is_stitch = False, is_flip = False,
                           scale_x = 1, scale_y = 1, rotate = 0, stitch_image_shape = None):
        """
        获得配准offset, 通过矩阵图和影像图的左上角00点

        Example:
        get_register_param(np.array([0, 0]), np.array([8198, 7297]), is_stitch=True, is_flip=False,
        scale_x=1.1557383123643845, scale_y=1.153519119475865, rotate=-1.5531499999999998,
        stitch_image_shape=[24071, 25128])

        Args:
            chip_lt_point: 芯片左上角00点 -- np.array -- shape==(2, )  # like this np.array([0, 0])
            image_lt_point: 影像图左上角00点 -- np.array -- shape==(2,)  # like this np.array([0, 0])
            is_stitch: stitch图 | transform图
            is_flip: 是否是未翻转图, 该处判定为上下翻转
            scale_x: ipr直接读取的参数
            scale_y: ipr直接读取的参数
            rotate: ipr直接读取的参数
            stitch_image_shape: 拼接图尺寸（h, w）
        Return:
            offset: 最后transform图的offset -- [x, y]
        """
        scale_x = 1 / scale_x
        scale_y = 1 / scale_y
        rotate = -rotate
        image_lt_point = np.array([image_lt_point])
        if is_stitch:
            mat_scale = np.mat([[scale_x, 0, 0],
                                [0, scale_y, 0],
                                [0, 0, 1]])
            mat_rotate = np.mat([[np.cos(np.radians(rotate)), -np.sin(np.radians(rotate)), 0],
                                 [np.sin(np.radians(rotate)), np.cos(np.radians(rotate)), 0],
                                 [0, 0, 1]])
            mat_center_f = np.mat([[1, 0, -stitch_image_shape[1] / 2],
                                   [0, 1, -stitch_image_shape[0] / 2],
                                   [0, 0, 1]])
            mat_center_s = np.mat([[1, 0, stitch_image_shape[1] / 2],
                                   [0, 1, stitch_image_shape[0] / 2],
                                   [0, 0, 1]])
            mat = mat_center_s * mat_scale * mat_rotate * mat_center_f

            _points = np.array([[0, 0],
                                [0, stitch_image_shape[0]],
                                [stitch_image_shape[1], 0],
                                [stitch_image_shape[1], stitch_image_shape[0]]])

            result = mat[:2, :] @ np.concatenate([_points, np.ones((_points.shape[0], 1))], axis=1).transpose(1, 0)
            x = result[0, :].max() - result[0, :].min()
            y = result[1, :].max() - result[1, :].min()

            mat_offset = np.mat([[1, 0, x / 2],
                                 [0, 1, y / 2],
                                 [0, 0, 1]])
            new_mat = mat_offset * mat_scale * mat_rotate * mat_center_f

            new_image_lt_point = \
                new_mat[:2, :] @ np.concatenate([image_lt_point,
                                                 np.ones((image_lt_point.shape[0], 1))], axis=1).transpose(1, 0)
            new_image_lt_point = np.array(new_image_lt_point).flatten()

            if not is_flip:
                new_image_lt_point[1] = y - new_image_lt_point[1]

            offset = chip_lt_point - new_image_lt_point
        else:
            offset = chip_lt_point - image_lt_point

        return offset


TRACK_POINTS_S13 = "S13_track_points.txt"
TRACK_POINTS_S6 = "S6_track_points.txt"
ORI_S6 = "ori.S6.6.8fov.txt"
ORI_S13 = "ori.S13.6.8fov.txt"

FOV2CHIP_ROW_S6_OLD = {'F': 3.4, 'E': 10.2, 'D': 17, 'C': 23.8, 'B': 30.6, 'A': 37.4}
FOV2CHIP_ROW_S6_NEW = {'G': 3.4, 'F': 10.2, 'E': 17, 'D': 23.8, 'C': 30.6, 'A': 37.4}
FOV2CHIP_COL_S6 = {'1': 3.4, '2': 10.2, '3': 17, '4': 23.8, '5': 30.6, '6': 37.4}

FOV2CHIP_ROW_S13 = {'P': 3.4, 'N': 10.2, 'M': 17, 'L': 23.8, 'K': 30.6, 'J': 37.4,
                    'H': 44.2, 'G': 51, 'F': 57.8, 'E': 64.6, 'D': 71.4, 'C': 78.2, 'A': 85}
FOV2CHIP_COL_S13 = {'1': 3.4, '2': 10.2, '3': 17, '4': 23.8, '5': 30.6, '6': 37.4,
                    '7': 44.2, '8': 51, '9': 57.8, 'A': 64.6, 'C': 71.4, 'D': 78.2, 'E': 85}

MASK_FOV_LEN = 1470
FOV_LEN = 0


class ChipCoord:
    """

    """
    def __init__(self):
        pass

    @staticmethod
    def paint_points_image(output_path,
                           fov_x_min, fov_x_max,
                           fov_y_min, fov_y_max,
                           mask_x_min, mask_x_max,
                           mask_y_min, mask_y_max,
                           x_new_mask, y_new_mask,
                           x_zero_list, y_zero_list):
        """
        画图
        """

        plt.figure(figsize = (10, 10))
        # 与表达矩阵方向相反时：
        # plt.gca().invert_yaxis()
        plt.axis('off')
        # 画芯片区域内00点
        plt.vlines([fov_x_min, fov_x_max], ymin = fov_y_min, ymax = fov_y_max, linewidth = 3)
        plt.hlines([fov_y_min, fov_y_max], xmin = fov_x_min, xmax = fov_x_max, linewidth = 3)
        plt.vlines([mask_x_min, mask_x_max], ymin = -mask_y_max, ymax = -mask_y_min, linewidth = 3, colors = 'red')
        plt.hlines([-mask_y_max, -mask_y_min], xmin = mask_x_min, xmax = mask_x_max, linewidth = 3, colors = 'red')
        spot_120 = pd.DataFrame(list(product(x_zero_list, y_zero_list)), columns = ['x', 'y'])
        plt.scatter(spot_120['x'], spot_120['y'])
        plt.scatter(x_zero_list[0], y_zero_list[-1])
        # 画track线
        plt.vlines(x_new_mask, ymin = -mask_y_max, ymax = -mask_y_min, linewidth = 0.5)
        plt.hlines(y_new_mask, xmin = mask_x_min, xmax = mask_x_max, linewidth = 0.5)
        plt.savefig(os.path.join(output_path, "chip_info.png"), bbox_inches = 'tight')

    @staticmethod
    def _get_zero_points(x_list, y_list, x_min, y_min):
        """
        内部值固定 无需更改
        """
        #  00点 x
        j = 0
        x_zero = []
        for i in x_list:
            if i - j == 210 or i - j == 105 or i - x_min == 105:
                x_zero.append(i)
            j = i

        #  00点 y
        j = 0
        y_zero = []
        for i in y_list:
            if i - j == 210 or i - j == 105 or i - y_min == -105:
                y_zero.append(j)
            j = i

        return x_zero, y_zero

    @staticmethod
    def _get_ori_data(file_path, is_s13 = False):
        """

        """
        if is_s13:
            _file = ORI_S13
        else:
            _file = ORI_S6

        ori_data = pd.read_csv(os.path.join(file_path, _file), sep = '\t', keep_default_na = False)

        ori_data_c = ori_data.copy(deep = True)
        s1_name = ori_data_c['name']
        for _name in s1_name:
            ul_x = float(ori_data_c[ori_data_c['name'] == _name]['ul_x'].iloc[0])
            br_x = float(ori_data_c[ori_data_c['name'] == _name]['br_x'].iloc[0])
            ul_y = float(ori_data_c[ori_data_c['name'] == _name]['ul_y'].iloc[0])
            br_y = float(ori_data_c[ori_data_c['name'] == _name]['br_y'].iloc[0])

            mid_x = (ul_x + br_x) / 2
            mid_y = (ul_y + br_y) / 2

            ori_data_c.loc[len(ori_data_c.index)] = [_name + '11', ul_x, mid_y, mid_x, br_y]
            ori_data_c.loc[len(ori_data_c.index)] = [_name + '12', mid_x, mid_y, br_x, br_y]
            ori_data_c.loc[len(ori_data_c.index)] = [_name + '13', ul_x, ul_y, mid_x, mid_y]
            ori_data_c.loc[len(ori_data_c.index)] = [_name + '14', mid_x, ul_y, br_x, mid_y]

        return ori_data_c

    @staticmethod
    def _get_track_points_data(file_path):
        track_points_data = pd.read_csv(os.path.join(file_path, TRACK_POINTS_S13),
                                        sep = '\t', names = ['x', 'y'])

        return track_points_data

    def get_zero_point_info(self,
                            chip_name: str,
                            mask_cut_info: str,
                            file_path: str = None,
                            paint_track_image: bool = False,
                            output_path: str = None,
                            old_chip: bool = False) -> Union[None, Tuple[int, int, float]]:
        """
        Args:
            chip_name: str --
            mask_cut_info: str -- "20-28_33-41"
            file_path: str -- file dir path
            paint_track_image: bool --
            output_path: str -- save chip image
            old_chip: bool -- old version chip, default False

        """

        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), "chip_file")

        if len(chip_name) > 10:  # 不考虑长码芯片
            return 3150, 3150, 0  # 默认值

        # 判断S13
        if chip_name[0] == "Y":
            is_s13 = True
            _fcr = FOV2CHIP_ROW_S13
            _fcc = FOV2CHIP_COL_S13
        else:
            is_s13 = False
            if old_chip: _fcr = FOV2CHIP_ROW_S6_OLD
            else: _fcr = FOV2CHIP_ROW_S6_NEW
            _fcc = FOV2CHIP_COL_S6

        # 小芯片
        if chip_name[-2:].isdigit(): is_small_chip = True
        else: is_small_chip = False

        track_points_data = self._get_track_points_data(file_path)
        ori_data = self._get_ori_data(file_path, is_s13)

        info = re.split('-|_', mask_cut_info)
        row = [int(info[2]), int(info[3])]
        col = [int(info[0]), int(info[1])]

        try:
            if not is_small_chip:
                for k, v in _fcr.items():
                    if row[0] <= v <= row[1]: row_name = k

                for k, v in _fcc.items():
                    if col[0] <= v <= col[1]: col_name = k

                area = row_name + col_name
                if old_chip:
                    if not is_s13:  # S6老芯片规则
                        if area[0] != "A":
                            area[0] = chr(ord(area[0]) + 1)
                    else:  # S13老芯片规则母鸡
                        pass

                # TODO 芯片命名纠错
                if area != chip_name[-2:]:
                    pass

            else:
                # 小芯片不纠了 规则太乱
                area = chip_name[-4:]

        except Exception as e:
            clog.info(f"Chip name not find, return default param.")
            return 3150, 3150, 0  # 默认值

        if area not in list(ori_data['name']):
            return 3150, 3150, 0  # 默认值

        #  all chip points
        mask_x_min = (col[0] - 1) * MASK_FOV_LEN
        mask_x_max = col[1] * MASK_FOV_LEN
        mask_y_min = (row[0] - 1) * MASK_FOV_LEN
        mask_y_max = row[1] * MASK_FOV_LEN

        _points_chip = track_points_data.loc[(track_points_data['x'] > mask_x_min) &
                                             (track_points_data['y'] > mask_y_min) &
                                             (track_points_data['x'] < mask_x_max) &
                                             (track_points_data['y'] < mask_y_max)]

        x_new_mask = list(set(list(_points_chip['x'])))
        y_new_mask = list(set(list(_points_chip['y'])))
        for k in range(len(y_new_mask)):
            y_new_mask[k] *= -1
        x_new_mask.sort()
        y_new_mask.sort()

        #  fov points
        fov_x_min = float(ori_data[ori_data['name'] == area]['ul_x'].iloc[0]) - FOV_LEN
        fov_x_max = float(ori_data[ori_data['name'] == area]['br_x'].iloc[0]) + FOV_LEN
        fov_y_min = -float(ori_data[ori_data['name'] == area]['ul_y'].iloc[0]) + FOV_LEN
        fov_y_max = -float(ori_data[ori_data['name'] == area]['br_y'].iloc[0]) - FOV_LEN
        _points_fov = track_points_data.loc[(track_points_data['x'] > fov_x_min) &
                                            (track_points_data['y'] > -fov_y_min) &
                                            (track_points_data['x'] < fov_x_max) &
                                            (track_points_data['y'] < -fov_y_max)]
        x_new_fov = list(set(list(_points_fov['x'])))
        y_new_fov = list(set(list(_points_fov['y'])))
        for k in range(len(y_new_fov)):
            y_new_fov[k] *= -1
        x_new_fov.sort()
        y_new_fov.sort()

        x_zero_list, y_zero_list = self._get_zero_points(x_new_fov, y_new_fov, fov_x_min, fov_y_min)

        if paint_track_image:
            self.paint_points_image(output_path,
                                    fov_x_min, fov_x_max,
                                    fov_y_min, fov_y_max,
                                    mask_x_min, mask_x_max,
                                    mask_y_min, mask_y_max,
                                    x_new_mask, y_new_mask,
                                    x_zero_list, y_zero_list)

        gene_x = (x_zero_list[0] - mask_x_min) * 2
        gene_y = (-mask_y_min - y_zero_list[-1]) * 2

        chip_x = (x_zero_list[0] - fov_x_min) * 2
        chip_y = (fov_y_min - y_zero_list[-1]) * 2

        chip_dis = np.sqrt(chip_x ** 2 + chip_y ** 2)

        return gene_x, gene_y, chip_dis


@njit(parallel=True)
def multiply_sum(regist_img, vis_img):
    """
    2023/09/20 @fxzhao 使用numba加速计算
    """
    h, w = regist_img.shape
    line_sums = np.zeros(h)
    for i in prange(h):
        line_sum = 0
        for j in range(w):
            if regist_img[i][j] > 0:
                line_sum += vis_img[i][j]
        line_sums[i] = line_sum
    return np.sum(line_sums)


if __name__ == '__main__':
    # cc = ChipCoord()
    #
    # gene_x, gene_y, chip_dis = cc.get_zero_point_info("SS200000059_NC", "76-83_1-8")

    rbnc = RegistrationByNanChip()

    offset = rbnc.get_register_param(np.array([0, 0]), np.array([1000, 9000]), is_stitch=True, is_flip = False,
        scale_x=1, scale_y=1, rotate= 45,
        stitch_image_shape=[10000, 10000])

    import json
    from glob import glob
    import os
    import tifffile
    import cv2
    import numpy as np

    vipshome = r'C:\vips-dev-8.12\bin'
    os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

    regist_path = r"D:\Data\qc\new_qc_test_data\regist_issue\A02177C4"
    ipr_path = glob(os.path.join(regist_path, "**.ipr"))[0]
    with h5py.File(ipr_path, "r") as f:
        # json_obj = json.load(f)
        scale_x = f["Register"].attrs["ScaleX"]
        scale_y = f["Register"].attrs["ScaleY"]
        rotation = f["Register"].attrs["Rotation"]
        # chip_template = f["ChipInfo"]["FOVTrackTemplate"]
        # offset_ori = f["AnalysisInfo"]["input_dct"]["offset"]
        # rot_ori = f["AnalysisInfo"]["input_dct"]["rot_type"]
    # fov_transformed_path = os.path.join(regist_path, '4_register', 'fov_stitched_transformed.tif')
    # fov_transformed = tifffile.imread(fov_transformed_path)
    chip_template = [[240, 300, 330, 390, 390, 330, 300, 240, 420], [240, 300, 330, 390, 390, 330, 300, 240, 420]]
    fov_stitched_path = glob(os.path.join(regist_path, '**fov_stitched.tif'))[0]
    fov_stitched = tifffile.imread(fov_stitched_path)

    # czi mouse brain -> stitch shape (2, x, x)
    if len(fov_stitched.shape) == 3:
        fov_stitched = fov_stitched[0, :, :]

    # try:
    #     gene_exp_path = glob(os.path.join(regist_path, "**raw.tif"))[0]
    # except IndexError:
    #     try:
    #         gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**_gene_exp.tif"))[0]
    #     except IndexError:
    #         gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**.gem.tif"))[0]

    gene_exp_path = glob(os.path.join(regist_path, "**gene.tif"))[0]
    gene_exp = cv2.imread(gene_exp_path, -1)

    track_template = np.loadtxt(glob(os.path.join(regist_path, '**template.txt'))[0])  # stitch template
    flip = True
    # im_shape = np.loadtxt(os.path.join(regist_path, "4_register", "im_shape.txt"))
    rg = Registration()
    rg.mass_registration_stitch(
        fov_stitched,
        gene_exp,
        chip_template,
        track_template,
        scale_x,
        scale_y,
        rotation,
        flip
    )
    print(rg.offset, rg.rot90, rg.score)
    rg.transform_to_regist()
    regist_img = rg.regist_img
    print("asd")
    tifffile.imwrite(os.path.join(regist_path, "new_regist_1.tif"), regist_img)
