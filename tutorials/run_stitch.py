import os

import pandas as pd
import numpy as np
import tifffile
import h5py

from cellbin.modules.stitching import Stitching
from cellbin.utils.file_manager import search_files, rc_key


def filename2index(file_name, style='motic', row_len=None):
    file_name = os.path.basename(file_name)
    if style.lower() in ['motic', 'cghd']:
        tags = os.path.splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
        x_str = xy[0]
        y_str = xy[1]
        return [int(y_str), int(x_str)]
    elif style.lower() == 'zeiss':
        line = os.path.splitext(file_name)[0].split('_')
        c = int(float(line[2]))
        r = int(float(line[1]))
        return [c, r]
    elif style.lower() == "leica dm6b":
        num = file_name.split("_")[1][1:]
        x = int(int(num) / row_len)
        y = int(int(num) % row_len)
        if x % 2 == 1:
            y = row_len - y - 1
        return [y, x]
    else:
        return None

def location2npy(xlsx: str):
    def get_npy(arr):
        h, w = arr.shape
        h_ = int(h / 2)
        mat = np.zeros((h_ - 1, w - 1, 2), dtype=np.int64)
        mat[:, :, 0] = arr[1:h_, 1:]
        mat[:, :, 1] = arr[h_ + 1:, 1:]
        return mat

    table_name = 'location'
    pr = pd.read_excel(xlsx, sheet_name=[table_name], header=None)
    location = get_npy(np.array(pr[table_name]))
    return location

def imagespath2dict(images_path, style='zeiss', row_len=None):
    image_support = ['.jpg', '.png', '.tif', '.tiff']
    fov_images = search_files(images_path, exts=image_support)
    src_fovs = dict()
    for it in fov_images:
        col, row = filename2index(it, style=style, row_len=row_len)
        src_fovs[rc_key(row, col)] = it

    return src_fovs


def test_stitching(image_src, rows, cols, chip_name, location=None, output=None):
    """
    Args:
        image_src: {'r_c': path, ...}
        rows:
        cols:
        location:
        output_path:
    """
    stitch = Stitching()
    stitch.set_size(rows, cols)
    if location is not None:
        stitch.set_global_location(location)
    stitch.stitch(image_src, output)
    image = stitch.get_image()
    tifffile.imwrite(os.path.join(output, f'{chip_name}.tif'), image)


def demo_stio(scope_src, chip_name, output):
    from stio.microscopy.slide_factory import MicroscopeBaseFileFactory
    from stio import slide2ipr0d0d1
    msp = MicroscopeBaseFileFactory().create_microscope_file(scope_src)
    ipr_manager = slide2ipr0d0d1(msp)
    fov_location = ipr_manager.stitch.scope_stitch.global_loc
    image_root = scope_src
    image_dict = ipr_manager.research.fovs_tag
    if len(ipr_manager.research.fovs_tag) != 0:
        image_dict = dict()
        for row, col in np.ndindex(ipr_manager.research.fovs_tag.shape):
            path = ipr_manager.research.fovs_tag[row, col]
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            image_dict[rc_key(row, col)] = os.path.join(image_root, path)
    rows = fov_location.shape[0]
    cols = fov_location.shape[1]
    loc = fov_location

    test_stitching(image_dict, rows, cols, chip_name, loc, output)


if __name__ == "__main__":
    # ipr_path = r''
    # src = r''
    # output = r''
    # # ipr_path = r''
    # # src = r''
    # # output = r''
    # #
    # # test_stitching(image_dict, rows, cols, chip_name, loc, output)
    # scope_src = r"D:\Data\630_result_check\clarity_issue\images"
    # chip_name = "B01615B4"
    # output = r"D:\Data\630_result_check\clarity_issue"
    # demo_stio(scope_src, chip_name, output)
    # scope_src = r"D:\Data\630_result_check\clarity_issue\images"
    # chip_name = "B01615B4"
    # output = r"D:\Data\630_result_check\clarity_issue"
    # demo_stio(scope_src, chip_name, output)
    images_path = r"D:\02.data\raozuming\zeiss_6_6\SS200000753BR_A1F6_20230805_215425\SS200000753BR_A1F6"
    # excel_path = r"D:\02.data\raozuming\zeiss_6_6\SS200000753BR_A1F6_20230805_215425\SS200000753BR_A1F6_SC_20230805_215425_2.2.0_dev.xlsx"
    # loc = location2npy(excel_path)
    src_fovs = imagespath2dict(images_path)
    test_stitching(src_fovs, rows=41, cols=41, chip_name="SS200000753BR_A1F6", location=None,
                   output=r"D:\02.data\raozuming\zeiss_6_6\SS200000753BR_A1F6_20230805_215425")