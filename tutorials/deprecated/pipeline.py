import os
from time import time
import h5py
import numpy as np
import argparse
from shutil import copyfile
import json
from datetime import datetime
import math
from json import loads, dumps, dump
import tarfile
from glob import glob
import cv2
import pandas as pd
import gzip
import shutil
from importlib.metadata import version
import traceback
import multiprocessing as mp

from cellbin.modules.stitching import Stitching
from cellbin.modules.registration import Registration
from cellbin.modules.tissue_segmentation import TissueSegmentation
from cellbin.modules.cell_segmentation import CellSegmentation
from cellbin.modules.cell_labelling import CellLabelling
from cellbin.utils import clog
from cellbin.image import Image
from cellbin.dnn.weights import auto_download_weights
from cellbin.image.augmentation import clarity_enhance_method
from cellbin.utils.file_manager import search_files, rc_key
from cellbin.image.augmentation import f_resize
from cellbin.image.augmentation import f_ij_16_to_8
from cellbin.image.augmentation import f_gray2bgr
from cellbin.image.augmentation import f_ij_auto_contrast
from cellbin.image.augmentation import f_rgb2gray
from cellbin.modules.iqc.clarity_qc import ClarityQC
from cellbin.image.mask import iou
from cellbin import WEIGHT_DIR, PIPELINE_CONFIG_PATH

from stio.chip import STOmicsChip
from stio.matrix_loader import MatrixLoader

# Constant
RPI_VERSION = '0.0.2'
PROG_VERSION = 'SAP'
JIQUN_CF = "/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/st_ster/pipeline_config.json"
JIQUN_ZOO = "/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/st_ster/cellbin_weights"
IMAGE_SUPPORT = ['.jpg', '.png', '.tif', '.tiff']

# Default
DEFAULT_CELL_TYPE = "CELL"
DEFAULT_CONFIG_PATH = PIPELINE_CONFIG_PATH
DEFAULT_WEIGHTS_DIR = WEIGHT_DIR
DEFAULT_VERSION = "SAP"


def generate_gem(mask_paths, gem_path, out_path):
    def row(gem_path):
        if gem_path.endswith('.gz'):
            with gzip.open(gem_path, 'rb') as f:
                first_line = bytes.decode(f.readline())
                if '#' in first_line:
                    rows = 6
                else:
                    rows = 0
        else:
            with open(gem_path, 'rb') as f:
                first_line = bytes.decode(f.readline())
                if '#' in first_line:
                    rows = 6
                else:
                    rows = 0
        return rows

    clog.info("Reading data..")
    gem = pd.read_csv(gem_path, sep='\t', skiprows=row(gem_path))
    assert "MIDCount" in gem.columns
    gem['x'] -= gem['x'].min()
    gem['y'] -= gem['y'].min()

    for mp in mask_paths:
        mask_path = mp
        filename = mask_path.replace('\\', '/').split('/')[-1].split('.')[0]
        i = Image()
        i.read(mask_path)
        mask = i.image
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)
        _, maskImg = cv2.connectedComponents(mask, connectivity=8)
        cur_gem = gem.copy(deep=True)
        cur_gem['CellID'] = maskImg[cur_gem['y'], cur_gem['x']]

        cell_gem = os.path.join(out_path, f'{filename}.gem')

        cur_gem.to_csv(cell_gem, sep='\t', index=False)
        # os.system('gzip {}'.format(os.path.join(out_path, f'{filename}.gem')))
        with open(cell_gem, 'rb') as f_in:
            with gzip.open(cell_gem + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(cell_gem)


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


def file_rc_index(file_name):
    tags = os.path.split(file_name)[1].split('_')
    xy = list()
    for tag in tags:
        if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
    c = xy[0]
    r = xy[1]
    return [int(r), int(c)]


def imagespath2dict(images_path):
    fov_images = search_files(images_path, exts=IMAGE_SUPPORT)
    src_fovs = dict()
    for it in fov_images:
        col, row = file_rc_index(it)
        src_fovs[rc_key(row, col)] = it

    return src_fovs


def outline(image, line_width):
    import cv2 as cv
    image = np.where(image != 0, 1, 0).astype(np.uint8)
    edge = np.zeros((image.shape), dtype=np.uint8)
    contours, hierachy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    r = cv.drawContours(edge, contours, -1, (255, 255, 255), line_width)
    return r


def _write_attrs(gp, d):
    """ Write dict to hdf5.Group as attributes. """
    for k, v in d.items():
        gp.attrs[k] = v


def json_serialize(obj, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as fd:
        str_dct = dumps(obj, default=lambda o: o.__dict__)
        dump(loads(str_dct), fd, indent=2, ensure_ascii=False)


def splitImage(im, tag, imgSize, h5_path, bin_size):
    """ Split image into patches with imgSize and save to h5 file. """
    # get number of patches
    height, width = im.shape[:2]
    # num_x = int(width/imgSize) + 1
    # num_y = int(height/imgSize) + 1
    num_x = math.ceil(width / imgSize)
    num_y = math.ceil(height / imgSize)

    with h5py.File(h5_path, 'a') as out:
        group = out.require_group(f'{tag}/bin_{bin_size}')

        # write attributes
        attrs = {'sizex': width,
                 'sizey': height,
                 'XimageNumber': num_x,
                 'YimageNumber': num_y}
        _write_attrs(group, attrs)

        # write dataset
        for x in range(0, num_x):
            for y in range(0, num_y):
                # deal with last row/column images
                x_end = min(((x + 1) * imgSize), width)
                y_end = min(((y + 1) * imgSize), height)
                if im.ndim == 3:
                    small_im = im[y * imgSize:y_end, x * imgSize:x_end, :]
                else:
                    small_im = im[y * imgSize:y_end, x * imgSize:x_end]

                data_name = f'{x}/{y}'
                # h_, w_ = small_im.shape
                # if h_ * w_ == 0: continue
                try:
                    # normal dataset creation
                    group.create_dataset(data_name, data=small_im)
                except Exception as e:
                    # if dataset already exists, replace it with new data
                    del group[data_name]
                    group.create_dataset(data_name, data=small_im)


def createPyramid(imgs, h5_path, imgSize=256, x_start=0, y_start=0, mag=(2, 10, 50, 100, 15), s_type='ssDNA'):
    """ Create image pyramid and save to h5. """
    img = imgs['ssDNA']
    # get height and width
    height, width = img.shape[:2]
    # im = np.rot90(im, 1)  ## 旋转图片，配准后的图片应该不用旋转了

    # write image metadata
    with h5py.File(h5_path, 'a') as h5_out:
        meta_group = h5_out.require_group('metaInfo')
        info = {'imgSize': imgSize,
                'x_start': x_start,
                'y_start': y_start,
                'sizex': width,
                'sizey': height,
                'version': '0.0.1'}
        _write_attrs(meta_group, info)

    # write image pyramid of bin size
    for k, img_p in imgs.items():
        # if img.dtype != 'uint8':
        #     img_p = transfer_16bit_to_8bit(img_p)
        for bin_size in mag:
            if k == 'CellMask':
                img_p = outline(img_p, line_width=2)
            if k == 'TissueMask':
                img_p = outline(img_p, line_width=100)
            im_downsample = img_p[::bin_size, ::bin_size]
            splitImage(im_downsample, k, imgSize, h5_path, bin_size)


class Pipeline(object):
    def __init__(self):
        self._ipr_path = None
        self._gem_path = None

        self.t_seg = None
        self.c_seg = None
        self.clarity_eval = None
        self.stitcher = Stitching()
        self.registrator = Registration()
        self.chip_template_getter = STOmicsChip()
        self.bmr = binary_mask_rle()

        self._input_path = None
        self.image_paths = None
        self._output_path = None
        self.is_stitched = None

        self._stereo_chip = None
        self._stain_type = None
        self._cell_seg_type = None
        self.chip_name = None
        self.config_file = None
        self.zoo_dir = None
        self.c_config = None
        self.t_config = None
        self.clarity_config = None
        self.stitch_config = None
        self.regist_config = None
        self.weight_names = None
        self.x_start = None
        self.y_start = None

        self._stitch_img = None
        self._regist_img = None
        # self._tissue_img = None
        # self._cell_img = None
        # self.log_file = None

        self.mask = None

        # self.file_name = None
        self.version = None
        self.debug_mode = False  # TODO: 后面这个需要加个接口! @dzh

        # 命名
        self.stitch_img_name = 'fov_stitched'
        self.stitch_template = 'stitch_template'
        self.gene_template = 'matrix_template'
        self.stitch_transform_name = 'fov_stitched_transformed'
        self.transform_template = 'transform_template'
        self.regist_img_name = 'regist'
        self.gene_img_name = 'gene'
        self.tissue_mask_name = 'tissue_cut'
        self.cell_mask_name = 'mask'
        self.cell_correct_mask_name = 'cell_mask_correct'
        self.clarity_eval_name = 'clarity_eval'
        self.filtered_cell_mask_name = 'cell_mask_filter'
        self.filtered_cell_correct_mask_name = 'cell_mask_correct_filter'

        self.img_ext = '.tif'
        self.txt_ext = '.txt'

        # time cost
        self.stitch_time = 0
        self.regist_time = 0
        self.tissue_seg_time = 0
        self.cell_seg_time = 0
        self.cell_seg_correct_time = 0

        # default init
        self.set_cell_seg_type(DEFAULT_CELL_TYPE)
        self.set_config_file(DEFAULT_CONFIG_PATH)
        self.set_zoo_dir(DEFAULT_WEIGHTS_DIR)
        self.set_version(DEFAULT_VERSION)

        clog.info("Init success.")

    def set_cell_seg_type(self, ct):
        self._cell_seg_type = ct

    def set_config_file(self, cf):
        with open(cf, 'r') as f:
            th_dict = json.load(f)
        self.config_file = th_dict
        self.c_config = self.config_file['cell_seg']
        self.t_config = self.config_file['tissue_seg']
        self.clarity_config = self.config_file['clarity_eval']
        self.stitch_config = self.config_file['stitch']
        self.regist_config = self.config_file.get('registration', {})
        cpu_count = mp.cpu_count()
        self.stitch_config['running_config']['num_threads'] = min(cpu_count // 2,
                                                                  self.stitch_config['running_config']['num_threads'])
        self.config_file['cell_correct']['num_threads'] = min(cpu_count // 2,
                                                              self.config_file['cell_correct']['num_threads'])
        clog.info(f"Using threads for stitch  {self.stitch_config['running_config']['num_threads']}\n"
                  f"using threads for  cell correct {self.config_file['cell_correct']['num_threads']}")
        print("asd")

    def set_zoo_dir(self, path: str):
        self.zoo_dir = path

    def set_version(self, v: str):
        self.version = v

    def auto_load_weights(self, ):
        self.weight_names = [
            self.c_config[self._cell_seg_type][self._stain_type],
            self.t_config[self._stain_type],
        ]
        auto_download_weights(self.zoo_dir, self.weight_names)  # if empty, auto download weights

    def prepare_input(self):
        clog.info(f"Preparing input for pipeline (images)")
        fname, ext = os.path.splitext(self._input_path)
        if ext == '.gz':
            # 解压文件 （延用开发版本）
            with tarfile.open(self._input_path, 'r') as tfo:
                for tarinfo in tfo:
                    if tarinfo.name == '':
                        continue
                    if os.path.splitext(tarinfo.name)[1] == '' or os.path.splitext(tarinfo.name)[1] == '.tif' or \
                            os.path.splitext(tarinfo.name)[1] == '.czi' or os.path.splitext(tarinfo.name)[1] == '.tiff':
                        tfo.extract(tarinfo, self._output_path)
            if not self.is_stitched:
                # 找到小图路径
                for name in os.listdir(self._output_path):
                    cur_path = os.path.join(self._output_path, name)
                    clog.info(f"Current path -> {cur_path}")
                    if os.path.isdir(cur_path):
                        img_dir = cur_path
                        break
                else:
                    raise Exception(f"Could not find fov directory")

            else:
                # 找到大图路径
                for f in os.listdir(self._output_path):
                    large_path = os.path.join(self._output_path, f)
                    if large_path.endswith('.tif'):
                        img_dir = large_path

        elif ext == '':  # 小图路径
            img_dir = self._input_path
        elif ext in IMAGE_SUPPORT:  # 单张大图
            self.is_stitched = True
            img_dir = self._input_path
        clog.info(f"Source image path(dir) {img_dir}")
        if not self.is_stitched:
            self.image_paths = imagespath2dict(img_dir)
        else:
            self.image_paths = img_dir

    def initialize_clarity_eval(self):
        try:
            clog.info('Initialize for clarity eval')
            self.clarity_eval = ClarityQC()
            self.clarity_eval.load_model(
                model_path=os.path.join(self.zoo_dir, self.clarity_config[self._stain_type]),
                batch_size=self.clarity_config['running_config']['batch_size'],
                gpu=self.clarity_config['running_config']['gpu']
            )
            clog.info('Load weights (Clarity Eval-{}) finished.'.format(self._stain_type))
        except Exception as e:
            clog.error(traceback.format_exc())

    def initialize_t_seg(self):
        try:
            clog.info('Initialize for tissue seg')
            clog.info(f'Tissue seg weight: {os.path.join(self.zoo_dir, self.t_config[self._stain_type])}')
            self.t_seg = TissueSegmentation(
                model_path=os.path.join(self.zoo_dir, self.t_config[self._stain_type]),
                stype=self._stain_type,
                gpu=self.t_config['running_config']['gpu'],
                num_threads=self.t_config['running_config']['num_threads']
            )
            clog.info('Load weights (tissue-{}) finished.'.format(self._stain_type))
        except Exception as e:
            clog.error(traceback.format_exc())

    def initialize_c_seg(self):
        try:
            clog.info('Initialize for cell seg')
            self.c_seg = CellSegmentation(
                model_path=os.path.join(self.zoo_dir, self.c_config[self._cell_seg_type][self._stain_type]),
                gpu=self.c_config["running_config"]["gpu"],
                num_threads=self.c_config['running_config']['num_threads']
            )
            clog.info('Load weights (cell-{}-{}) finished.'.format(self._cell_seg_type, self._stain_type))
        except Exception as e:
            clog.error(traceback.format_exc())

    def load_initial_info(self, ):
        try:
            clog.info('Load initial info (StainType, ChipSN...)')
            with h5py.File(self._ipr_path, 'r') as conf:
                self._stain_type = conf['QCInfo'].attrs['StainType']
                self._stain_type = self._stain_type.upper()
                self.chip_name = conf['ImageInfo'].attrs['STOmicsChipSN']
                if not self.chip_template_getter.is_chip_number_legal(self.chip_name):
                    raise Exception(f"{self.chip_name} not supported")
                    # 获取芯片号对应的芯片模板
                short_sn = self.chip_template_getter.get_valid_chip_no(self.chip_name)
                self._stereo_chip = self.chip_template_getter.get_chip_grids(short_sn)
                self.is_stitched = conf['ImageInfo'].attrs['StitchedImage']
        except Exception as e:
            clog.error(traceback.format_exc())

    def initial_ipr_attrs(self, ):
        try:
            clog.info('Initial some ipr attrs and datasets for pipeline (TissueSeg, CellSeg, ManualState...)')
            with h5py.File(self._ipr_path, 'a') as conf:
                if 'StereoResepVersion' not in conf['ImageInfo'].attrs.keys():
                    conf['ImageInfo'].attrs['StereoResepVersion'] = PROG_VERSION
                if "IPRVersion" not in conf.keys():
                    conf.attrs['IPRVersion'] = '0.0.1'
                if "StereoResepSwitch" not in conf.keys():
                    group = conf.create_group('StereoResepSwitch')
                    group.attrs['stitch'] = True
                    group.attrs['tissueseg'] = True
                    group.attrs['cellseg'] = True
                    group.attrs['register'] = True
                if "ManualState" not in conf.keys():
                    group = conf.create_group('ManualState')
                    group.attrs['stitch'] = False
                    group.attrs['tissueseg'] = False
                    group.attrs['cellseg'] = False
                    group.attrs['register'] = False
                if 'TissueSeg' not in conf.keys():
                    conf.create_group('TissueSeg')
                if 'CellSeg' not in conf.keys():
                    conf.create_group('CellSeg')
        except Exception as e:
            clog.error(traceback.format_exc())

    def _stitching(self, ):
        clog.info('Stitching')
        try:
            # TODO 替代io模块, ipr临时测通, 不涉及写入
            with h5py.File(self._ipr_path, 'r') as conf:
                research_stitch_module = conf['Research']['Stitch']
                location = research_stitch_module['StitchFovLocation'][...]
                track_template = research_stitch_module['GlobalTemplate'][...]
                microscope_stitch = research_stitch_module.attrs['MicroscopeStitch']
                if microscope_stitch == 0:
                    h_jitter = research_stitch_module['HorizontalJitter'][...]
                    v_jitter = research_stitch_module['VerticalJitter'][...]

                rows = conf['ImageInfo'].attrs['ScanRows']
                cols = conf['ImageInfo'].attrs['ScanCols']

            fov_stitched_path = os.path.join(self._output_path, self.stitch_img_name + self.img_ext)
            #########
            if not self.is_stitched:
                self.stitcher.set_size(rows, cols)
                self.stitcher.set_global_location(location)
                self.stitcher.stitch(src_fovs=self.image_paths, output_path=self._output_path)

                try:
                    fft_detect_channel = self.stitch_config['fft_channel'].get(self._stain_type, 0)
                except Exception as e:
                    fft_detect_channel = 0

                clog.info(f'Stitch module fft channel: {fft_detect_channel}')
                # TODO 得到行向和列向拼接偏移量评估
                if microscope_stitch == 0:
                    self.stitcher.set_jitter(h_jitter, v_jitter)
                    x_jitter_eval, y_jitter_eval = self.stitcher._get_stitch_eval()
                else:
                    self.stitcher._get_jitter(
                        src_fovs=self.image_paths,
                        process=self.stitch_config['running_config']['num_threads'],
                        fft_channel=fft_detect_channel
                    )
                    x_jitter_eval, y_jitter_eval = self.stitcher._get_jitter_eval()

                self._stitch_img = self.stitcher.get_image()
                Image.write_s(self._stitch_img, fov_stitched_path,
                              compression=True)

                with h5py.File(self._ipr_path, 'a') as conf:
                    # 适配stio ipr，image studio ipr
                    stitch_module = conf['Stitch']
                    stitch_module.attrs['StitchingScore'] = -1
                    bgi_stitch_module_name = 'BGIStitch'
                    if bgi_stitch_module_name not in stitch_module.keys():
                        stitch_module.require_group(bgi_stitch_module_name)
                    bgi_stitch_module = stitch_module[bgi_stitch_module_name]
                    try:
                        bgi_stitch_module.attrs['StitchedGlobalHeight'] = self._stitch_img.shape[0]
                        bgi_stitch_module.attrs['StitchedGlobalWidth'] = self._stitch_img.shape[1]
                        if 'StitchedGlobalLoc' in bgi_stitch_module.keys():
                            del bgi_stitch_module['StitchedGlobalLoc']
                    except:
                        pass
                    bgi_stitch_module.create_dataset('StitchedGlobalLoc', data=np.array(location))
                    stitch_eval_module_name = 'StitchEval'
                    if stitch_eval_module_name not in stitch_module.keys():
                        stitch_module.require_group(stitch_eval_module_name)
                    stitch_eval_module = stitch_module[stitch_eval_module_name]
                    if 'StitchEvalH' in stitch_eval_module.keys():
                        del stitch_eval_module['StitchEvalH']
                        del stitch_eval_module['StitchEvalV']
                    stitch_eval_module.create_dataset('StitchEvalH', data=x_jitter_eval)
                    stitch_eval_module.create_dataset('StitchEvalV', data=y_jitter_eval)
                    conf['StereoResepSwitch'].attrs['stitch'] = False
            else:
                try:
                    copyfile(self.image_paths, fov_stitched_path)  # TODO: 处理大图！！
                except Exception as e:
                    clog.error(traceback.format_exc())

            np.savetxt(os.path.join(self._output_path, self.stitch_template + self.txt_ext), track_template)
            clog.info('Stitching Done')

        except Exception as e:
            clog.error(traceback.format_exc())

    def _registration(self, ):
        clog.info('Registration')
        try:
            # TODO 替代io模块, ipr临时测通, 不涉及写入
            with h5py.File(self._ipr_path, 'r') as conf:
                scale_x = conf['Register'].attrs['ScaleX']
                scale_y = conf['Register'].attrs['ScaleY']
                rotate = conf['Register'].attrs['Rotation']
                track_template = conf['Research']['Stitch']['GlobalTemplate'][...]
            rgb_image = None  # grayscale image is None, rgb image is not None
            if self._stitch_img is None:
                i = Image()
                stitch_img_path = os.path.join(self._output_path, self.stitch_img_name + self.img_ext)
                i.read(image=stitch_img_path)
                self._stitch_img = i.image

            if self._stitch_img.ndim == 3:
                rgb_image = self._stitch_img.copy()
                regist_channel = self.regist_config.get('channel', {}).get('HE', None)
                if regist_channel is None:
                    if self._stain_type == 'HE':
                        clog.info(f"Stitch image is {self._stitch_img.ndim} channel, stain type is HE and regist "
                                  f"channel is None, convert, using cell seg rgb2gray method")
                        self._stitch_img = f_rgb2gray(self._stitch_img, need_not=True)
                    else:
                        clog.info(f"Stitch image is {self._stitch_img.ndim} channel and regist channel is None,"
                                  f"convert using regular rgb2gray method")
                        self._stitch_img = f_rgb2gray(self._stitch_img)
                else:
                    clog.info(f"Stitch image is {self._stitch_img.ndim} channel and regist channel is {regist_channel},"
                              f"using {regist_channel} as regist channel")
                    self._stitch_img = self._stitch_img[:, :, regist_channel]
            else:
                clog.info(f"Stitch image is {self._stitch_img.ndim} channel, no need to convert")

            # _, _, x_start, y_start, gene_exp = gef2image(self._gem_path, self._output_path)
            ml = MatrixLoader(self._gem_path, self._output_path)
            gene_exp, self.x_start, self.y_start = ml.f_gene2img_pd()

            #########
            self.registrator.mass_registration_stitch(
                fov_stitched=self._stitch_img,
                vision_image=gene_exp,
                chip_template=self._stereo_chip,
                track_template=track_template,
                scale_x=1 / scale_x,
                scale_y=1 / scale_y,
                rotation=rotate,
                flip=True
            )
            vision_cp = self.registrator.vision_cp
            self.registrator.transform_to_regist()
            self._regist_img = self.registrator.regist_img  # TODO 配准图返回
            regist_img_copy = self._regist_img.copy()
            # regist_score = self.registrator.register_score(regist_img_copy, gene_exp)  # 会改变第一个入参
            if rgb_image is not None:
                clog.info(f"Stitch image is rgb, generating rgb regist image")
                fov_transform = self.registrator.stitch_to_transform(
                    fov_stitch=rgb_image,
                    scale_x=1 / scale_x,
                    scale_y=1 / scale_y,
                    rotation=rotate,
                )
                self.registrator.fov_transformed = fov_transform
                self.registrator.transform_to_regist()
                self._regist_img = self.registrator.regist_img  # 配准图返回

            np.savetxt(os.path.join(self._output_path, self.gene_template + self.txt_ext), vision_cp)
            regist_path = os.path.join(self._output_path, f'{self.chip_name}_{self.regist_img_name}' + self.img_ext)
            Image.write_s(self._regist_img, regist_path, compression=True)
            Image.write_s(
                gene_exp,
                os.path.join(self._output_path, f'{self.chip_name}_{self.gene_img_name}' + self.img_ext),
                compression=True)

            ####
            # 适配流程
            h_, w_ = self.registrator.fov_transformed.shape[:2]
            json_serialize({'height': h_, 'width': w_}, os.path.join(self._output_path, 'attrs.json'))
            if w_ > h_:
                thumb = f_resize(f_ij_16_to_8(self.registrator.fov_transformed.copy()), (1500, int(h_ * 1500 / w_)))
            else:
                thumb = f_resize(f_ij_16_to_8(self.registrator.fov_transformed.copy()), (int(w_ * 1500 / h_), 1500))
            Image.write_s(thumb, os.path.join(self._output_path, 'transform_thumb.png'))

            # 组织最大外接矩阵
            bbox = [0, 0, self._regist_img.shape[1], self._regist_img.shape[0]]
            with open(os.path.join(self._output_path, f'{self.chip_name}_tissue_bbox.csv'), 'w') as f:
                f.write('left\tupper\tright\tlower\n')
                f.write('\t'.join(list(map(str, bbox))))
            ####

            Image.write_s(
                self.registrator.fov_transformed,
                os.path.join(self._output_path, self.stitch_transform_name + self.img_ext),
                compression=True)
            np.savetxt(os.path.join(self._output_path, self.transform_template + self.txt_ext),
                       self.registrator.adjusted_stitch_template_unflip)
            with h5py.File(self._ipr_path, 'a') as conf:
                if 'MatrixTemplate' in conf['Register'].keys():
                    del conf['Register']['MatrixTemplate']
                if 'TransformTemplate' in conf['Stitch'].keys():
                    del conf['Stitch/TransformTemplate']
                conf['Stitch'].create_dataset(
                    'TransformTemplate',
                    data=self.registrator.adjusted_stitch_template_unflip[:, :2]
                )
                conf['Register'].create_dataset('MatrixTemplate', data=vision_cp[:, :2], compression='gzip')
                conf['Register'].attrs['CounterRot90'] = self.registrator.rot90
                conf['Register'].attrs['OffsetX'] = self.registrator.offset[0]
                conf['Register'].attrs['OffsetY'] = self.registrator.offset[1]
                # conf['Register'].attrs['RegisterScore'] = regist_score
                conf['Register'].attrs['MatrixShape'] = gene_exp.shape
                conf['Register'].attrs['XStart'] = self.x_start
                conf['Register'].attrs['YStart'] = self.y_start
                conf['Register'].attrs['Flip'] = True
                conf['StereoResepSwitch'].attrs['register'] = False

            del self._stitch_img
            del gene_exp
            del self.registrator.fov_transformed
            del self._regist_img
            self._regist_img = None
            clog.info('Registration Done')

        except Exception as e:
            clog.error(traceback.format_exc())

    def get_regist_score(self):
        try:
            irt = Image()
            irt.read(os.path.join(self._output_path, f'{self.chip_name}_tissue_cut.tif'))
            tissue_mask = irt.image
            irc = Image()
            irc.read(os.path.join(self._output_path, f'{self.chip_name}_{self.gene_img_name}' + self.img_ext))
            gene_image = irc.image
            regist_score = self.registrator.register_score(tissue_mask, gene_image)
            clog.info(f"Regist score is: {regist_score}")
            with h5py.File(self._ipr_path, 'a') as conf:
                conf['Register'].attrs['RegisterScore'] = regist_score
            clog.info("Finish calculating regist score")
        except Exception as e:
            clog.error(traceback.format_exc())

    def get_stitch_score(self):
        try:
            stitch_result_dict = {}
            with h5py.File(self._ipr_path, 'a') as conf:
                try:
                    StitchEvalH = conf['Stitch']['StitchEval']['StitchEvalH']
                    stitchevalH = np.array(StitchEvalH)
                    stitch_result_dict['H_mean'] = np.mean(stitchevalH[stitchevalH != -1])
                    stitch_result_dict['H_max'] = np.abs(stitchevalH).max()
                except:
                    stitch_result_dict['H_mean'] = -1
                    stitch_result_dict['H_max'] = -1
                try:
                    StitchEvalV = conf['Stitch']['StitchEval']['StitchEvalV']
                    stitchevalV = np.array(StitchEvalV)
                    stitch_result_dict['V_mean'] = np.mean(stitchevalV[stitchevalV != -1])
                    stitch_result_dict['V_max'] = np.abs(stitchevalV).max()
                except:
                    stitch_result_dict['V_mean'] = -1
                    stitch_result_dict['V_max'] = -1
                conf['Stitch']['StitchEval'].attrs['StitchScore'] = "_".join(map(str, stitch_result_dict.values()))
            clog.info("Finish calculating stitch score")
        except Exception as e:
            clog.error(traceback.format_exc())

    def get_cell_seg_score(self):
        def check_wh(wh, bin):
            cel_num = np.subtract(np.float32(wh), np.multiply(np.ones(wh.shape, dtype=np.float32), bin))
            cel_num_01 = np.multiply(cel_num[:, 0], cel_num[:, 1]) <= 0
            cel_num_11 = np.bitwise_and(cel_num[:, 0] > 0, cel_num[:, 1] > 0)
            return np.sum(np.bitwise_or(cel_num_01, cel_num_11))

        try:
            cell_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.cell_mask_name}' + self.img_ext
            )
            i = Image()
            i.read(cell_mask_path)
            cell_mask = i.image
            mask = np.clip(cell_mask, 0, 1)
            all_cell_area = np.sum(mask)
            cell_num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cell_num -= 1
            stats = stats[1:, 2:5]
            wh = stats[:, 0:2]
            area = stats[:, 2]

            cell_num_40 = check_wh(wh, 40)
            result = [all_cell_area / cell_num, cell_num_40 / cell_num]

            with h5py.File(self._ipr_path, 'a') as conf:
                cell_seg = conf['Research'].require_group('CellSeg')
                cell_seg.attrs['CellSegScore'] = "_".join(map(str, result))
        except Exception as e:
            clog.error(traceback.format_exc())

    def _tissue_segmentation(self):
        clog.info('Tissue Segmentation')
        try:
            if self._regist_img is None:
                regist_path = os.path.join(self._output_path, f'{self.chip_name}_{self.regist_img_name}' + self.img_ext)
                i = Image()
                i.read(image=regist_path)
                self._regist_img = i.image

            mask = self.t_seg.run(self._regist_img)
            Image.write_s(
                mask,
                os.path.join(self._output_path, f'{self.chip_name}_{self.tissue_mask_name}' + self.img_ext),
                compression=True
            )
            with h5py.File(self._ipr_path, 'a') as conf:
                tissue_module = 'TissueSeg'
                if tissue_module not in conf.keys():
                    conf.require_group(tissue_module)
                conf[tissue_module].attrs['TissueSegShape'] = mask.shape
                conf['StereoResepSwitch'].attrs['tissueseg'] = False
            del mask
            clog.info('Tissue Done')
        except Exception as e:
            clog.error(traceback.format_exc())

    def _clarity_eval(self):
        clog.info('Clarity Eval')
        try:
            if self._regist_img is None:
                regist_path = os.path.join(self._output_path, f'{self.chip_name}_{self.regist_img_name}' + self.img_ext)
                i = Image()
                i.read(image=regist_path)
                self._regist_img = i.image
            self.clarity_eval.set_enhance_func(clarity_enhance_method.get(self._stain_type, None))
            self.clarity_eval.run(
                img=self._regist_img,
            )
            self.clarity_eval.post_process()
            clarity_mask = self.clarity_eval.black_img
            Image.write_s(
                clarity_mask,
                os.path.join(self._output_path, f"{self.chip_name}_clarity_mask" + self.img_ext),
                compression=True
            )
            clarity_heatmap = self.clarity_eval.draw_img
            cv2.imwrite(
                os.path.join(self._output_path, f"{self.chip_name}_{self.clarity_eval_name}" + self.img_ext),
                clarity_heatmap,
            )
            clog.info('Clarity Eval Finished')
        except Exception as e:
            clog.error(traceback.format_exc())

    def _cell_segmentation(self):
        clog.info('Cell Segmentation')
        try:
            if self._regist_img is None:
                regist_path = os.path.join(self._output_path, f'{self.chip_name}_{self.regist_img_name}' + self.img_ext)
                i = Image()
                i.read(image=regist_path)
                self._regist_img = i.image
            self.mask = self.c_seg.run(self._regist_img)
            trace = self.c_seg.get_trace(self.mask)
            cell_mask_path = os.path.join(self._output_path, f'{self.chip_name}_{self.cell_mask_name}' + self.img_ext)
            Image.write_s(self.mask, cell_mask_path, compression=True)
            with h5py.File(self._ipr_path, 'a') as conf:
                cell_seg = conf['Research'].require_group('CellSeg')
                # cell_seg.attrs['CellMaskPath'] = cell_mask_path
                cell_seg.create_dataset('CellSegTrace', data=np.array(trace), compression='gzip')
                cell_seg_module = 'CellSeg'
                if cell_seg_module not in conf.keys():
                    conf.require_group(cell_seg_module)
                conf[cell_seg_module].attrs['CellSegShape'] = self.mask.shape
                conf['StereoResepSwitch'].attrs['cellseg'] = False
            del self.c_seg
            clog.info('Cell Segmentation Done')
        except Exception as e:
            clog.error(traceback.format_exc())

    def _cell_labeling(self):
        clog.info('Cell Labeling')
        try:
            if self.mask is None:
                cell_mask_path = os.path.join(
                    self._output_path,
                    f'{self.chip_name}_{self.cell_mask_name}' + self.img_ext
                )
                i = Image()
                i.read(image=cell_mask_path)
                mask = i.image
            else:
                mask = self.mask
            ml = MatrixLoader(self._gem_path)
            if self._gem_path.lower().endswith(('.bgef', '.gef')):
                new_file = os.path.join(self._output_path, f"{self.chip_name}_exp.gem")
                ml.bgef2gem(bgef_file=self._gem_path, gem_file=new_file, binsize=1)
            else:
                # gem gz
                new_file = self._gem_path

            # 修正
            cl = CellLabelling(
                mask,
                new_file
            )
            cl.set_process(self.config_file["cell_correct"]['num_threads'])
            correct_mask, self.exp_matrix = cl.run_fast(distance=10)

            # mask写出
            # correct_mask = cl.draw_corrected_mask(correct_mask)
            correct_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.cell_correct_mask_name}' + self.img_ext
            )
            Image.write_s(correct_mask, correct_mask_path, compression=True)
        except Exception as e:
            clog.error(traceback.format_exc())

    def _iqc(self, ):
        clog.info('Image QC')
        pass

    def _check_qc_flag(self, ):

        with h5py.File(self._ipr_path) as conf:
            qc_flag = conf['QCInfo'].attrs['QCPassFlag']

        return True if qc_flag == 1 else False

    def save_preview(self):
        try:
            ir = Image()
            ir.read(glob(os.path.join(self._output_path, '*_transformed.tif'))[0])
            img = ir.image
            if img.dtype != 'uint8':
                img = f_ij_16_to_8(img)
            irt = Image()
            irt.read(glob(os.path.join(self._output_path, '*_tissue_cut.tif'))[0])
            tissue = irt.image
            if tissue.dtype != 'uint8':
                tissue = f_ij_16_to_8(tissue)
            tissue[tissue > 0] = 255
            img = f_resize(img)
            h, w = img.shape[:2]
            mask = f_resize(tissue, (w, h))
            if img.ndim == 2:
                img = f_gray2bgr(img)
            contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
            img = f_ij_auto_contrast(img)
            with h5py.File(self._ipr_path, 'r+') as conf:
                if 'Preview' in conf.keys():
                    del conf['Preview']
                conf.create_dataset('Preview', data=img, compression='gzip')
            clog.info('Success to create preview')
        except Exception as e:
            clog.error(traceback.format_exc())

    def cell_mask_post_process(self):
        """
        Based on cell mask and cell correct mask
        Generate
        - filtered cell mask
        - filtered cell correct mask
        """
        try:
            clog.info("Filtering cell mask and correct mask based on tissue mask")
            cell_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.cell_mask_name}' + self.img_ext
            )
            correct_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.cell_correct_mask_name}' + self.img_ext
            )
            tissue_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.tissue_mask_name}' + self.img_ext
            )
            i = Image()
            i.read(cell_mask_path)
            cell_mask = i.image
            i.read(correct_mask_path)
            correct_mask = i.image
            i.read(tissue_mask_path)
            tissue_mask = i.image
            filtered_cell_mask = cell_mask * tissue_mask
            filtered_cell_correct_mask = correct_mask * tissue_mask
            i.write_s(
                image=filtered_cell_mask,
                output_path=os.path.join(
                    self._output_path,
                    f'{self.chip_name}_{self.filtered_cell_mask_name}' + self.img_ext
                ),
                compression=True
            )
            i.write_s(
                image=filtered_cell_correct_mask,
                output_path=os.path.join(
                    self._output_path,
                    f'{self.chip_name}_{self.filtered_cell_correct_mask_name}' + self.img_ext
                ),
                compression=True
            )
            clog.info("Finished filtering")
        except Exception as e:
            clog.error(traceback.format_exc())

    def gem_file_post_process(self):
        """
        Based on these four mask and input gene matrix (gem file)
        Generate corresponding four gem file
        """
        try:
            clog.info("Generating new gene files based on different masks")
            masks = [
                self.cell_mask_name,
                self.cell_correct_mask_name,
                self.filtered_cell_mask_name,
                self.filtered_cell_correct_mask_name
            ]
            if self._gem_path.lower().endswith(('.bgef', '.gef')):
                gem_path = os.path.join(self._output_path, f"{self.chip_name}_exp.gem")
                if not os.path.exists(gem_path):
                    ml = MatrixLoader(self._gem_path)
                    ml.bgef2gem(bgef_file=self._gem_path, gem_file=gem_path, binsize=1)
            else:
                gem_path = self._gem_path

            mask_paths = []
            for mask_name in masks:
                cur_mask_path = os.path.join(
                    self._output_path,
                    f'{self.chip_name}_{mask_name}' + self.img_ext
                )
                mask_paths.append(cur_mask_path)
            generate_gem(
                mask_paths=mask_paths,
                gem_path=gem_path,
                out_path=self._output_path
            )
            clog.info("Finished generating new gene files")
        except Exception as e:
            clog.error(str(e))

    def run_x_bin(self, gem_path, input_path, ipr_path, output=None):
        clog.info(f"Using version {self.version}")
        self._gem_path = gem_path
        self._input_path = input_path
        self._output_path = os.path.join(output, "registration")
        os.makedirs(self._output_path, exist_ok=True)
        clog.info(f"Pipeline output dir {self._output_path}")
        ipr_name = os.path.split(ipr_path)[-1]
        self._ipr_path = os.path.join(self._output_path, ipr_name)
        copyfile(ipr_path, self._ipr_path)
        if self._check_qc_flag():

            clog.info(f"Pipeline using ipr {self._ipr_path}")
            self.load_initial_info()  # 初始化多个模块需要用到的信息from ipr，主要是读取
            self.initial_ipr_attrs()  # 初始化pipeline需要用到的一些group，如果不用stio的话是需要pipeline自己生成的
            self.prepare_input()  # 解压缩qc的targz，找到图片路径
            clog.info('Start pipline')

            if self.config_file["operation"]["Stitching"]:
                start = time()
                self._stitching()
                end = time()
                self.stitch_time = end - start
                clog.info(f"Stitching module time cost: {self.stitch_time}")
                self.get_stitch_score()
            if self.config_file["operation"]["Register"]:
                start = time()
                self._registration()
                end = time()
                self.regist_time = end - start
                clog.info(f"Registration module time cost: {self.regist_time}")
            # 初始化分割模型
            if self.config_file["operation"]["Tissue_Segment"]:
                auto_download_weights(self.zoo_dir, [self.t_config[self._stain_type]])
                start = time()
                self.initialize_t_seg()
                self._tissue_segmentation()
                end = time()
                self.tissue_seg_time = end - start
                clog.info(f"Tissue seg module time cost: {self.tissue_seg_time}")
                auto_download_weights(self.zoo_dir, [self.clarity_config[self._stain_type]])
                self.initialize_clarity_eval()
                self._clarity_eval()
                self._get_tissue_seg_score()
                self.get_regist_score()
                self.save_preview()
            if self.config_file["operation"]["Cell_Segment"]:
                auto_download_weights(self.zoo_dir, [self.c_config[self._cell_seg_type][self._stain_type]])
                start = time()
                self.initialize_c_seg()
                self._cell_segmentation()
                end = time()
                self.cell_seg_time = end - start
                clog.info(f"Cell seg module time cost: {self.cell_seg_time}")
                self.get_cell_seg_score()
            if self.config_file["operation"]["Tissue_Segment"] and self.config_file["operation"]["Cell_Segment"]:
                self.regist_to_rpi()
            if self.config_file["operation"]["Cell_Correct"]:
                start = time()
                self._cell_labeling()
                end = time()
                self.cell_seg_correct_time = end - start
                clog.info(f"Cell seg correct time cost: {self.cell_seg_correct_time}")
            if self.config_file["operation"]["Cell_Segment"] and self.config_file["operation"]["Cell_Correct"] \
                    and self.version.upper() == 'SAP':
                self.cell_mask_post_process()
                self.gem_file_post_process()
            self.write_time_cost()
        else:
            clog.info('QC failed, skip pipeline.')

    def write_time_cost(self):
        with h5py.File(self._ipr_path, 'a') as f:
            research_info = f['Research']
            research_info.attrs['StitchTime'] = self.stitch_time
            research_info.attrs['RegistTime'] = self.regist_time
            research_info.attrs['TissueSegTime'] = self.tissue_seg_time
            research_info.attrs['CellSegTime'] = self.cell_seg_time
            research_info.attrs['CellSegCorrectTime'] = self.cell_seg_correct_time

    def encode_mask(self, mask):
        encode_mask = self.bmr.encode(mask)
        return encode_mask

    def _get_tissue_seg_score(self):
        try:
            irt = Image()
            irt.read(os.path.join(self._output_path, f'{self.chip_name}_{self.tissue_mask_name}' + self.img_ext))
            tissue_mask = irt.image
            irc = Image()
            irc.read(os.path.join(self._output_path, f"{self.chip_name}_clarity_mask" + self.img_ext))
            clarity_mask = irc.image
            iou_result = iou(clarity_mask, tissue_mask)
            iou_result = round(iou_result, 3) * 100
            clog.info(f"Tissue seg score {iou_result}")
            with h5py.File(self._ipr_path, 'a') as conf:
                tissue_module = 'TissueSeg'
                if tissue_module not in conf.keys():
                    conf.require_group(tissue_module)
                conf[tissue_module].attrs['TissueSegScore'] = iou_result
            clog.info("Finish calculating tissue seg score")
        except Exception as e:
            clog.error(traceback.format_exc())

    @staticmethod
    def transform_to_rpi(transform_path, output_path, chip_name, img_size=256, x_start=0, y_start=0,
                         mag=(2, 10, 50, 100, 150)):
        import gc
        def find_thresh(img: np.ndarray):
            """
            :param img: gray image
            :type img: numpy nd array
            :return: hmin, hmax
            :rtype: int, int
            """
            if not isinstance(img, np.ndarray):
                raise Exception(f"Input must be numpy nd array")
            input_dims = len(img.shape)
            if input_dims != 2:
                raise Exception(f"Input should be 2 dimensional array, but input array is {input_dims} dimensions")

            # Constants
            limit = img.size / 10
            threshold = img.size / 5000
            n_bins = 256
            if img.dtype != 'uint8':
                bit_max = 65536
            else:
                bit_max = 256

            hist_min = np.min(img)
            hist_max = np.max(img)

            bin_size = (hist_max - hist_min) / n_bins
            hist, bins = np.histogram(img.flatten(), n_bins, [hist_min, hist_max])

            hmin = 0
            hmax = bit_max - 1

            for i in range(1, len(hist) - 1):
                count = hist[i]
                if count > limit:
                    continue
                if count > threshold:
                    hmin = i
                    break
            for i in range(len(hist) - 1, 0, -1):
                count = hist[i]
                if count > limit:
                    continue
                if count > threshold:
                    hmax = i
                    break

            hmin = hist_min + hmin * bin_size
            hmax = hist_min + hmax * bin_size

            hmin, hmax = int(hmin), int(hmax)

            hmin = max(0, hmin)
            hmax = min(bit_max - 1, hmax)

            if hmax > hmin:
                return hmin, hmax
            else:
                return 0, 0

        def image_type_convert(img):
            if img.ndim == 2:
                arr = img
            else:
                if img.shape[0] in [1, 2, 3, 4]:
                    img = img.transpose((1, 2, 0))

                if img.shape[2] == 1:
                    arr = img[:, :, 0]
                elif img.shape[2] == 2:
                    max_0 = np.sum(img[:, :, 0])
                    max_1 = np.sum(img[:, :, 1])
                    index = 0 if max_0 > max_1 else 1
                    arr = img[:, :, index]
                else:
                    arr = img
            return arr

        img_path = transform_path
        trans_reader = Image()
        trans_reader.read(img_path)
        transform_image = trans_reader.image
        t1 = time()
        transform_rpi_path = os.path.join(output_path, f'{chip_name}_fov_transformed.rpi')
        # write image metadata
        with h5py.File(transform_rpi_path, 'a') as h5_out:
            clog.info("Create new rpi")
            # get height and width
            height, width = transform_image.shape[:2]
            meta_group = h5_out.require_group('metaInfo')
            info = {'version': RPI_VERSION,
                    'imgSize': img_size,
                    'x_start': x_start,
                    'y_start': y_start,
                    'sizex': width,
                    'sizey': height}
            _write_attrs(meta_group, info)
            gc.collect()
            h5_out.file.flush()

            # write image pyramid of bin size
            color = '(255, 255, 255)'
            transform_image = image_type_convert(transform_image)
            group_name = 'ssDNA/Image'

            # 给group name增加 Color, GrayLevelElbow(图像像素值拐点的下限值), MaxGrayLevel, TrackLayer 属性
            tag = group_name
            try:
                track_group = tag.split('/')[0]
            except:
                clog.warning('track group must exist.')
                track_group = ''
            try:
                key_group = tag.split('/')[1]
            except:
                key_group = 'Image'
                tag = f'{track_group}/{key_group}'
            group_ = h5_out.require_group(tag)
            if key_group == 'Image':
                group_.attrs['TrackLayer'] = True
                group_.attrs['GrayLevelElbow'] = 0
                if transform_image.ndim == 2:
                    min_gray_value, max_gray_value = find_thresh(transform_image)
                    group_.attrs['MinGrayLevel'] = min_gray_value
                    group_.attrs['MaxGrayLevel'] = max_gray_value
                else:
                    group_.attrs['MinGrayLevel'] = 0
                    group_.attrs['MaxGrayLevel'] = 0
            else:
                group_.attrs['TrackLayer'] = False
                group_.attrs['MinGrayLevel'] = 0
                group_.attrs['MaxGrayLevel'] = 255
                group_.attrs['GrayLevelElbow'] = 0
            group_.attrs['Color'] = color
            h5_out.file.flush()

        for bin_size in mag:
            if transform_image.ndim == 3:
                im_downsample = transform_image[::bin_size, ::bin_size, :]
            else:
                im_downsample = transform_image[::bin_size, ::bin_size]
            splitImage(h5_path=transform_rpi_path, im=im_downsample, tag=group_name, imgSize=img_size, bin_size=bin_size)

        t2 = time()
        clog.info(f"Save rpi: {t2 - t1:.2f} seconds.")

    def regist_to_rpi(self):
        clog.info("Saving tissue/cell mask to ipr, generate regist rpi")
        # need regist, tissue mask, cell mask
        irt = Image()
        irt.read(os.path.join(self._output_path, f'{self.chip_name}_{self.tissue_mask_name}' + self.img_ext))
        tissue_mask = irt.image
        irc = Image()
        irc.read(os.path.join(self._output_path, f'{self.chip_name}_{self.cell_mask_name}' + self.img_ext))
        cell_mask = irc.image
        irg = Image()
        irg.read(os.path.join(self._output_path, f'{self.chip_name}_{self.regist_img_name}' + self.img_ext))
        regist_tif = irg.image
        with h5py.File(self._ipr_path, 'r+') as f:
            if 'CellMask' in f['CellSeg'].keys():
                del f['CellSeg']['CellMask']
            if 'TissueMask' in f['TissueSeg'].keys():
                del f['TissueSeg']['TissueMask']
            f['TissueSeg'].create_dataset('TissueMask', data=self.encode_mask(tissue_mask), compression='gzip')
            f['CellSeg'].create_dataset('CellMask', data=self.encode_mask(cell_mask), compression='gzip')
        with h5py.File(self._ipr_path, 'r') as f:
            x_start = f['Register'].attrs['XStart']
            y_start = f['Register'].attrs['YStart']
        result = dict()
        if self._stain_type == 'HE':
            s_type = 'HE'
        else:
            s_type = 'ssDNA'
        result[f'{s_type}'] = regist_tif
        result['TissueMask'] = tissue_mask
        result['CellMask'] = cell_mask
        rpi_path = os.path.join(self._output_path, f'{self.chip_name}.rpi')
        createPyramid(result, rpi_path, x_start=x_start, y_start=y_start, s_type=s_type)
        clog.info("Finished regist to rpi")

    @staticmethod
    def get_time():
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return dt


def run_pipeline(input_, output, ipr_path, matrix, cell_type, config, zoo_dir, version='SAP'):
    clog.info("--------------------------------start--------------------------------")
    image_root = input_
    output = output
    ipr_path = ipr_path
    gem_file = matrix
    cell_type = cell_type.upper()

    p = Pipeline()
    p.set_cell_seg_type(cell_type)
    p.set_config_file(config)
    p.set_zoo_dir(zoo_dir)
    p.set_version(version)

    p.run_x_bin(gem_path=gem_file, input_path=image_root, ipr_path=ipr_path, output=output)
    clog.info("--------------------------------end--------------------------------")


def main(args, para):
    clog.log2file(
        out_dir=args.output,
    )
    clog.set_level()

    try:
        clog.info(f"Cellbin Version {version('cell-bin')}")
    except Exception:
        clog.error(traceback.format_exc())

    clog.info(args)
    try:
        run_pipeline(
            input_=args.input,
            output=args.output,
            ipr_path=args.ipr_path,
            matrix=args.matrix,
            cell_type=args.cell_type.upper(),
            config=args.config,
            zoo_dir=args.zoo_dir,
            version=args.version
        )
    except Exception:
        clog.error(traceback.format_exc())


def arg_parser():
    usage = '''
    python pipeline.py -i /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/images/C4 
    -m /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/SS200000464BL_C4.raw.gef 
    -r /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/SS200000464BL_C4_20230403_125857_0.1.ipr 
    -o /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/test_out 
    -cf /home/dengzhonghan/Desktop/code/cellbin/scripts/moudule/pipeline_config.json 
    -z /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/weights
    '''

    clog.info(PROG_VERSION)
    parser = argparse.ArgumentParser(usage=usage)
    # Must have
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="Tar file / Input image dir.")
    parser.add_argument("-m", "--matrix", action="store", dest="matrix", type=str, required=True,
                        help="Input gene matrix.")
    parser.add_argument("-r", "--ipr_path", action="store", dest="ipr_path", type=str, required=True, help="Ipr path.")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=True,
                        help="Result output dir.")

    # Optional
    parser.add_argument("-cf", "--config", action="store", dest="config", type=str, default=JIQUN_CF,
                        help="Config file (Json)")
    parser.add_argument("-z", "--zoo", action="store", dest="zoo_dir", type=str, default=JIQUN_ZOO,
                        help="DNN weights dir")

    parser.add_argument("-ct", "--cell_seg", action="store", dest="cell_type", type=str, default='CELL',
                        help="Cell Seg type")
    parser.add_argument("-v", "--prog_version", action="store", dest="version", type=str, default=PROG_VERSION,
                        help="SAP or UAT")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)


if __name__ == '__main__':
    arg_parser()
