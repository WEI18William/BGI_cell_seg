import os
import cv2
import h5py
import numpy as np
import datetime
from distutils.util import strtobool
from importlib.metadata import version
import traceback

from cellbin.utils import clog
from cellbin.modules.image_qc import ImageQualityControl
from cellbin.utils.file_manager import rc_key
from cellbin.utils.file_manager import rc_key_trans
from cellbin import WEIGHT_DIR, QC_CONFIG_PATH
from stio import slide2ipr0d0d1
from stio.microscopy.slide_factory import MicroscopeBaseFileFactory
from stio.chip import STOmicsChip

VERSION = '0.1'

# Default
DEFAULT_MAGNIFICATION = 10
DEFAULT_FOV_SIZE = (2000, 2000)
DEBUG_MODE = False


class binary_mask_rle(object):
    """
    ref.: https://en.wikipedia.org/wiki/Run-length_encoding
    """

    def __init__(self):
        pass

    def encode(self, binary_mask):
        '''
        binary_mask: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        binary_mask[binary_mask >= 1] = 1
        pixels = binary_mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        runs = np.reshape(runs, (-1, 2))
        return runs

    def decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        # s = mask_rle.split()
        # starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts = mask_rle[:, 0]
        lengths = mask_rle[:, 1]
        starts -= 1
        ends = starts + lengths
        binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            binary_mask[lo:hi] = 1
        return binary_mask.reshape(shape)


class QcRunner(object):
    def __init__(self):
        self.image_qc = ImageQualityControl()
        self.zoo_dir = ""

        # qc input
        self.image_root = None  # user input
        self.is_stitched = None  # ipr read
        self.stain_type = None  # user input
        self.stereo_chip = None  # ipr read
        self.magnification = None  # ipr read
        self.fov_location = None  # ipr read
        self.image_dict = None  # ipr read
        self.fov_size = None  # ipr read, (w, h)
        self.config_file = None  # qc config file

        # cio
        self.ipr_manager = None
        self.chip_name = None
        self.chip_template_getter = STOmicsChip()

        #
        self.debug = False
        self.save_dir = None
        self.ipr_save_path = None
        self.file_name = None

        # constant
        self.tissue_bin_threshold = 0.1

    def read_microscope(self, src_data):
        msp = MicroscopeBaseFileFactory().create_microscope_file(src_data)
        self.ipr_manager = slide2ipr0d0d1(msp)
        # 芯片号合法化校验
        if not self.chip_template_getter.is_chip_number_legal(self.chip_name):
            raise Exception(f"{self.chip_name} not supported")
        # 获取芯片号对应的芯片模板
        short_sn = self.chip_template_getter.get_valid_chip_no(self.chip_name)
        self.stereo_chip = self.chip_template_getter.get_chip_grids(short_sn)
        self.is_stitched = self.ipr_manager.image_info.stitched_image
        if not self.is_stitched:
            self.image_root = src_data
        else:
            self.image_root = os.path.dirname(src_data)
        self.magnification = self.ipr_manager.image_info.scan_objective
        self.fov_location = self.ipr_manager.stitch.scope_stitch.global_loc
        self.image_dict = self.ipr_manager.research.fovs_tag
        if len(self.ipr_manager.research.fovs_tag) != 0:
            self.image_dict = dict()
            for row, col in np.ndindex(self.ipr_manager.research.fovs_tag.shape):
                path = self.ipr_manager.research.fovs_tag[row, col]
                if isinstance(path, bytes):
                    path = path.decode('utf-8')
                self.image_dict[rc_key(row, col)] = os.path.join(self.image_root, path)
        self.fov_size = (self.ipr_manager.image_info.fov_width, self.ipr_manager.image_info.fov_height)

    def check_qc_input(self, ):
        for key, val in self.image_dict.items():
            if not os.path.exists(val):
                raise Exception(f"{val} does not exist")

        if len(self.stereo_chip) == 0 or len(self.stereo_chip[1]) != 9:
            raise Exception("Stereo chip error")
        if not self.is_stitched and (
                len(self.fov_location) == 0 or len(self.fov_location.shape) != 3 or self.fov_location.shape[-1] != 2):
            raise Exception("Fov location error")
        if not isinstance(self.magnification, int) and not isinstance(self.magnification, float):
            raise Exception("magnification error")
        if self.fov_size is None or self.image_root is None or self.is_stitched is None:
            return 1
        return 0

    def initialize_qc(self):
        if self.check_qc_input() != 0:
            raise Exception(f"QC input error")
        else:
            self.file_name = "_".join([self.chip_name, self.get_time_stamp(), VERSION])
            self.ipr_save_path = os.path.join(self.save_dir, self.file_name + ".ipr")
            self.ipr_manager.write(self.ipr_save_path)
        # qc 赋值
        self.image_qc.set_is_stitched(self.is_stitched)
        self.image_qc.set_stain_type(self.stain_type)
        self.image_qc.set_stereo_chip(self.stereo_chip)
        self.image_qc.set_fov_loc(self.fov_location)
        self.image_qc.set_fov_size(self.fov_size)
        self.image_qc.set_magnification(self.magnification)
        self.image_qc.initialize_json(self.config_file)
        self.image_qc.set_zoo_dir(self.zoo_dir)
        self.image_qc.auto_load_weights()
        initialize_code = self.image_qc.initialize()
        clog.info(f"Image qc initialization finished")
        return initialize_code

    def run_qc(self):
        if self.is_stitched:
            self.image_dict = os.path.basename(list(self.image_dict.values())[0])
        self.image_qc.run(
            image_root=self.image_root,
            image_map=self.image_dict
        )

        self.write_to_ipr()
        return 0

    def set_zoo_dir(self, path: str):
        self.zoo_dir = path

    def set_stain_type(self, stain_type: str):
        self.stain_type = stain_type.upper()

    def set_chip_name(self, chip_name: str):
        self.chip_name = chip_name

    def set_w_h(self, w, h):
        # 小图模式可以直接从显微镜配置文件读取
        if self.is_stitched:
            self.fov_size = (w, h)

    def set_magnification(self, m):
        self.magnification = m

    def set_config_file(self, c):
        self.config_file = c

    def set_debug_mode(self, d):
        self.debug = d
        self.image_qc.set_debug_mode(d)

    def set_save_file(self, f: str):
        self.save_dir = f

    def write_to_ipr(self, ):
        with h5py.File(self.ipr_save_path, 'a') as f:
            # same with old version
            qc_info = f["QCInfo"]
            qc_info.attrs['ClarityScore'] = self.image_qc.clarity_score
            qc_info.attrs['GoodFOVCount'] = self.image_qc.good_fov_count
            qc_info.attrs['QCPassFlag'] = 0  # 默认0
            if self.stain_type == 'HE':
                self.tissue_bin_threshold = 0.01  # TODO: HE tissue bin 阈值， DAPI还是用默认的0.1
            if self.image_qc.global_template_flag:
                if self.image_qc.template_qc_recall >= self.tissue_bin_threshold or self.image_qc.template_max >= self.tissue_bin_threshold:
                    qc_info.attrs['QCPassFlag'] = 1  # 大于tissue bin阈值置为1
            clog.info(f"QCPassFlag: {qc_info.attrs['QCPassFlag']}")
            qc_info.attrs['StainType'] = self.stain_type
            qc_info.attrs['TrackLineScore'] = self.image_qc.track_score
            if self.is_stitched:
                qc_info.attrs['TotalFOVCount'] = self.image_qc.total_fov_count
            cps = qc_info['CrossPoints']
            tps_copy = self.image_qc.track_pts
            if self.image_qc.line_qc_flag:
                del tps_copy[self.image_qc.track_line_best_match[0]]
                tps_copy[self.image_qc.track_line_best_match[0]] = [self.image_qc.track_line_best_match[1], -1]
            elif self.image_qc.stitch_fov_index != -1:  # TODO
                del tps_copy[self.image_qc.stitch_fov_index]
                tps_copy[self.image_qc.stitch_fov_index] = [self.image_qc.stitch_fov_best_point, -1]
            for row_column, points in tps_copy.items():
                row_column = '_'.join(map(str, rc_key_trans(row_column)))
                if points[0]:  # TODO
                    if self.image_qc.line_qc_flag and row_column == '_'.join(
                            map(str, rc_key_trans(self.image_qc.track_line_best_match[0]))):
                        cps.create_dataset(row_column, data=np.array(points[0]))
                    elif self.image_qc.stitch_fov_index != -1 and row_column == '_'.join(
                            map(str, rc_key_trans(self.image_qc.stitch_fov_index))):
                        cps.create_dataset(row_column, data=np.array(points[0]))
                    else:
                        pts = np.array(points[0])
                        empty = np.zeros((len(pts), 4))
                        empty[:, :2] = pts[:, :2]
                        cps.create_dataset(row_column, data=empty)
            regist_info = f['Register']
            regist_info.attrs['Rotation'] = self.image_qc.rotation
            regist_info.attrs['ScaleX'] = self.image_qc.scale_x
            regist_info.attrs['ScaleY'] = self.image_qc.scale_y
            stitch_info = f["Stitch"]
            if self.image_qc.line_qc_flag:  # TODO
                stitch_info.attrs['TemplateSource'] = '_'.join(map(str, rc_key_trans(self.image_qc.src_fov_info[0])))
            elif self.image_qc.stitch_fov_index != -1:  # TODO
                stitch_info.attrs['TemplateSource'] = '_'.join(map(str, rc_key_trans(self.image_qc.stitch_fov_index)))
            # 大图评估热图
            stitch_eval_module_name = 'StitchEval'
            if stitch_eval_module_name not in stitch_info.keys():
                stitch_info.require_group(stitch_eval_module_name)
            stitch_eval_module = stitch_info[stitch_eval_module_name]
            stitch_eval_module.create_dataset(
                'GlobalDeviation',
                data=self.image_qc.stitch_template_global,
            )
            image_info = f["ImageInfo"]
            image_info.attrs['FOVHeight'] = self.fov_size[1]
            image_info.attrs['FOVWidth'] = self.fov_size[0]
            image_info.attrs['QCResultFile'] = self.file_name
            image_info.attrs['STOmicsChipSN'] = self.chip_name

            # research
            research_info = f['Research']
            research_info.attrs['PrepareTime'] = self.image_qc.prepare_time
            research_info.attrs['TissueSegTime'] = self.image_qc.tissue_seg_time
            research_info.attrs['TrackDetectTime'] = self.image_qc.track_detect_time
            research_info.attrs['TrackLineTime'] = self.image_qc.track_line_time
            research_info.attrs['StitchQcTime'] = self.image_qc.stitch_qc_time
            research_info.attrs['ClarityQcTime'] = self.image_qc.clarity_qc_time

            # stitch

            stitch_research = research_info.require_group('Stitch')
            stitch_research.attrs['StitchDiffMax'] = self.image_qc.stitch_diff_max
            stitch_research.attrs['JitterDiffMax'] = self.image_qc.jitter_diff_max
            stitch_research.attrs['TemplateMax'] = self.image_qc.template_max
            stitch_research.attrs['TemplateMean'] = self.image_qc.template_mean
            stitch_research.attrs['TemplateStd'] = self.image_qc.template_std
            stitch_research.attrs['TemplatePrecision'] = self.image_qc.template_qc_precision
            stitch_research.attrs['TemplateRecall'] = self.image_qc.template_qc_recall
            stitch_research.create_dataset('GlobalTemplate', data=np.array(self.image_qc.global_template))
            stitch_research.create_dataset('StitchDiff', data=self.image_qc.stitch_diff)
            stitch_research.create_dataset('JitterDiff', data=self.image_qc.jitter_diff)
            stitch_research.create_dataset('HorizontalJitter', data=self.image_qc.jitter[0])
            stitch_research.create_dataset('VerticalJitter', data=self.image_qc.jitter[1])
            stitch_research.attrs['StitchRoiFov'] = self.image_qc.stitch_roi
            stitch_research.attrs['MicroscopeStitch'] = self.image_qc.microscope_stitch
            stitch_research.create_dataset('StitchFovLocation', data=self.image_qc.stitch_fov_loc)

            # tissue seg
            if self.image_qc.config.operation.tissue_segment:
                tissue_research = research_info.require_group('Tissue')
                tissue_research.create_dataset('TissueMaskFov', data=self.image_qc.fov_tissue_mask)
                tissue_research.create_dataset('TissueMaskY', data=binary_mask_rle().encode(self.image_qc.tissue_mask))
                tissue_research.attrs['ClarityBox'] = self.image_qc.box_mosaic
                tissue_research.attrs['TissueSegArea'] = self.image_qc.tissue_area

            # regist
            regist_research = research_info.require_group('Regist')
            regist_research.attrs['TrackLineScore'] = self.image_qc.track_line_score
            regist_research.create_dataset('TrackFovMask', data=self.image_qc.track_pt_fov_mask)
            if self.image_qc.line_qc_flag:
                # regist_best_track_line = regist_research.require_group('best_track_line')
                regist_research.attrs['ScaleX'] = self.image_qc.track_line_best_match[2]
                regist_research.attrs['ScaleY'] = self.image_qc.track_line_best_match[3]
                regist_research.attrs['Rotation'] = self.image_qc.track_line_best_match[4]
                # regist_research.create_dataset(
                #     self.image_qc.track_line_best_match[0],
                #     data=np.array(self.image_qc.track_line_best_match[1])
                # )

            # clarity
            if self.image_qc.config.operation.clarity_qc:
                if self.image_qc.config.operation.tissue_segment:
                    tissue_research.attrs['TissueSegScore'] = self.image_qc.tissue_mask_score
                clarity_research = research_info.require_group('Clarity')
                clarity_research.attrs['Counts'] = str(self.image_qc.clarity_counts)
                clarity_research.attrs['CutSize'] = self.image_qc.clarity_cut_size
                clarity_research.attrs['Overlap'] = self.image_qc.clarity_overlap
                cv2.imwrite(os.path.join(self.save_dir, f"{self.file_name}.tif"), self.image_qc.clarity_heatmap)
                clarity_research.create_dataset('ClarityArr', data=self.image_qc.clarity_preds_arr)
                self.image_qc.clarity_cluster.savefig(os.path.join(self.save_dir, f"{self.file_name}.png"))
                # clarity_research.create_dataset('clarity_heatmap', data=self.image_qc.clarity_heatmap)

    def get_time_stamp(self):
        times = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return times


def qc_setup(
        stain_type,
        src_data,
        chip_name,
        save_dir,
        fov_size=DEFAULT_FOV_SIZE,
        magnification=DEFAULT_MAGNIFICATION,
        debug_mode=DEBUG_MODE,
        zoo_dir=WEIGHT_DIR,
        config_path=QC_CONFIG_PATH
):
    # clog.cl.__init__()
    clog.log2file(
        out_dir=save_dir,
    )
    try:
        clog.info(f"Cellbin Version {version('cell-bin')}")
    except Exception as e:
        clog.error(traceback.format_exc())
    w, h = fov_size
    qc_runner = QcRunner()
    os.makedirs(save_dir, exist_ok=True)
    qc_runner.set_save_file(save_dir)  # required
    # qc_runner.download_and_check_weights(zoo_dir)  # required
    qc_runner.set_zoo_dir(zoo_dir)  # required
    qc_runner.set_chip_name(chip_name)  # required
    qc_runner.set_stain_type(stain_type)  # required
    qc_runner.set_config_file(config_path)
    qc_runner.read_microscope(src_data)  # required

    qc_runner.set_w_h(w, h)  # 要在读显微镜之后set fov size, 只有大图需要做这个
    qc_runner.set_magnification(magnification)  # motic可读到，czi不知道
    qc_runner.set_debug_mode(debug_mode)
    initialize_code = qc_runner.initialize_qc()
    return qc_runner, initialize_code


def main_qc(**kwargs):
    try:
        # w, h = fov_size
        # qc_runner = QcRunner()
        # os.makedirs(save_dir, exist_ok=True)
        # qc_runner.set_save_file(save_dir)  # required
        # # qc_runner.download_and_check_weights(zoo_dir)  # required
        # qc_runner.set_zoo_dir(zoo_dir)  # required
        # qc_runner.set_chip_name(chip_name)  # required
        # qc_runner.set_stain_type(stain_type)  # required
        # qc_runner.set_config_file(config_path)
        # qc_runner.read_microscope(src_data)  # required
        #
        # qc_runner.set_w_h(w, h)  # 要在读显微镜之后set fov size, 只有大图需要做这个
        # qc_runner.set_magnification(magnification)  # motic可读到，czi不知道
        # qc_runner.set_debug_mode(debug_mode)
        qc_runner, initialize_code = qc_setup(**kwargs)
        if initialize_code == 0:
            qc_runner.run_qc()
        else:
            pass
    except Exception as e:
        clog.error(traceback.format_exc())


def main():
    usage = """
    /hwfssz1/ST_BIOINTEL/P20Z10200N0039/07.software/anaconda3/envs/cellbin/bin/python /path/test_qc.py 
    -i input image folder/path
    -o output dir
    -c chip number
    -z /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/st_ster/cellbin_weights
    -d True
    -j /path/qc_config.json
    """
    import argparse
    ArgParser = argparse.ArgumentParser(usage=usage)
    ArgParser.add_argument("--version", action="version", version=VERSION)
    ArgParser.add_argument("-i", "--input", action="store", dest="input_path", type=str, required=True,
                           help="Image folder.")
    ArgParser.add_argument("-o", "--output", action="store", dest="output_path", type=str, required=True,
                           help="QC results output folder.")
    ArgParser.add_argument("-c", "--chip", action="store", dest="chip_name", type=str, required=True, help="Chip name")
    ArgParser.add_argument("-z", "--zoo", action="store", dest="zoo_dir", type=str, required=True,
                           default='', help="DNN weights dir")
    ArgParser.add_argument("-j", "--json", action="store", dest="json_path", type=str, required=True,
                           help="Config file path")
    ArgParser.add_argument("-s", "--stain", action="store", dest="stain_type", type=str, default='DAPI',
                           help="Stain type. (SSDNA, DAPI, CZI).")
    ArgParser.add_argument("-m", "--magnification", action="store", dest="magnification", type=int,
                           default=DEFAULT_MAGNIFICATION,
                           help="10X or 20X")
    ArgParser.add_argument("-f", "--fov_size", action="store", dest="fov_size", type=int, nargs='+',
                           default=DEFAULT_FOV_SIZE,
                           help="Fov size used for large image")
    ArgParser.add_argument("-d", "--debug", action="store", dest="debug", type=lambda x: bool(strtobool(x)),
                           default=DEBUG_MODE, help="Debug mode")

    (para, args) = ArgParser.parse_known_args()
    main_qc(
        stain_type=para.stain_type,
        src_data=para.input_path,
        chip_name=para.chip_name,
        fov_size=tuple(para.fov_size),
        magnification=para.magnification,
        save_dir=para.output_path,
        debug_mode=para.debug,
        zoo_dir=para.zoo_dir,
        config_path=para.json_path
    )


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn')
    main()
