import glob
import os
import sys
import tifffile
import argparse
import copy
from cellbin.modules.cell_segmentation import CellSegmentation, SUPPORTED_TYPES
from skimage.morphology import remove_small_objects
from cellbin.modules import StainType
import numpy as np
import pdb
from fnmatch import fnmatch
import cv2
import re


def get_filelist(dir, patterns=["*"], filelist=[]):
    new_dir = dir
    if os.path.isdir(dir):
        for s in os.listdir(dir):
            new_dir = os.path.join(dir, s)
            get_filelist(new_dir, patterns, filelist)
    else:
        for p in patterns:
            if fnmatch(dir, p):
                filelist.append(dir)
                break
    return filelist


def mask_to_outline(mask):
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, 255, 1)
    return out


WEIGHTS = {
    "CELLCUT": {
        "CELL": {
            "SSDNA": "cellseg_bcdu_SHDI_221008_tf.onnx",
            "DAPI": "cellseg_bcdu_SHDI_221008_tf.onnx",
            "HE": "cellseg_bcdu_SHDI_221008_tf.onnx",
            "RNA": "cellseg_unet_RNA_20230606.onnx"
        }
    },
}

MAIN_PATH = "/media/Data/dzh/weights"

def extract_identifier(file_path):
    # 从文件路径中提取文件名
    file_name = os.path.basename(file_path)
    # 使用正则表达式提取标识符
    match = re.match(r"^(.*?)(_after|_cut|_regist|\.tif)", file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"无法从文件名 {file_name} 中提取标识符")

def main():
    # parser = argparse.ArgumentParser(description="you should add those parameter")
    # parser.add_argument('-i', "--input", help="the input img path")
    # parser.add_argument('-o', "--output", help="the output file")
    # parser.add_argument("-g", "--gpu", help="the gpu index", default=-1)
    # parser.add_argument("-th", "--num_threads", help="num_threads", default="0")
    #
    # args = parser.parse_args()
    # input_path = args.input
    # output_path = args.output
    # gpu = args.gpu
    # num_threads = args.num_threads

    # if input_path is None or output_path is None:
    #     print("please check your parameters")
    #     sys.exit()
    # print(args)
    # model_path = r"D:\Data\qc\new_qc_test_data\clarity\bad\test_imgs\test_download\cellseg_bcdu_SHDI_221008_tf.onnx"

    # model_path = r"D:\code\public\cell_segmentation_v03\model\weight_cell_230529.onnx"
    # input_path_lst = ["/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/A02085D2_after_tc_regist.tif",
    #                   "/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/A02085D3_after_tc_regist.tif",
    #                   "/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/A02085F2_after_tc_regist.tif",
    #                   "/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/A02085D3_after_tc_regist.tif",
    #                   "/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/A02185C2_after_tc_regist.tif",
    #                   "/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/A02185D2_after_tc_regist.tif",
    #                   "/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/C02245A2_after_tc_regist.tif"]
    #input = ["/media/Data/dzh/data/cellbin/FF-HE-C-Seg-Upgrade/tc_162/C02245A2_after_tc_regist.tif"]
    input = "/media/Data/spx/train_mouse_brain_project/tc_cls/img/Mouse_brain/"
    input_path_lst = []
    if isinstance(input, str):
        # 如果 input 是一个字符串，将其作为目录路径处理
        input_dir = input
        for i in os.listdir(input_dir):
            file = os.path.join(input_dir, i)
            input_path_lst.append(file)
    elif isinstance(input, list):
        # 如果 input 是一个列表，将其直接赋值给 input_path_lst
        input_path_lst = input
    else:
        raise ValueError("Input must be either a string or a list.")

    out_path = '/media/Data/spx/train_mouse_brain_project/mouse_brain_8_9_out/Mouse_brain/'
    os.makedirs(out_path, exist_ok=True)
    gpu = "0"
    # img_type = StainType.ssDNA.value  # 目前细胞分割仅支持 ssdna, HE, rna, DAPI
    # img_type = StainType.HE.value
    img_type = StainType.HE.value
    num_threads = 0
    model_path = "/media/Data/spx/train_mouse_brain_project/train_he_162/weight_log/mouse_brain/models_Aug08_21-50-46/weights.26-0.12.onnx"
    cell_bcdu = CellSegmentation(
        model_path=model_path,
        gpu=gpu,
        num_threads=num_threads,
        img_type=img_type
    )
    # file_lst = glob.glob(os.path.join(input_path, "*.tif"))
    file_lst = []
    for p in input_path_lst:
        file_lst.extend(get_filelist(p, ["*.tif"], []))
    print(file_lst)
    for file in file_lst:
        # try:
        sn = extract_identifier(file)
        
        mask_name = f"{sn}_mask.tif"
        if os.path.exists(os.path.join(out_path, mask_name)):
            print(os.path.join(out_path, mask_name))
            continue

        img = tifffile.imread(file)
        # img = cv2.resize(src=img,dsize=None,fx=2.5,fy=2.5)
        # 返回的mask
        mask = cell_bcdu.run(img)

        mask[mask > 0] = 255
        # outline = mask_to_outline(mask)
        # #c3 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # c3 = img
        # c3[:, :, 1] = cv2.addWeighted(c3[:, :, 1], 1.0, outline, 0.5, 0)
        # tifffile.imwrite(os.path.join(out_path, os.path.splitext(name)[0]+"_outline.tif"), c3, compression='zlib')

        # mask = remove_small_objects(mask.astype(np.bool8), min_size=50).astype(np.uint8)
        # 返回的统计数据,box_h,box_w,area
        # trace = CellSegmentation.get_trace(mask)
        tifffile.imwrite(os.path.join(out_path, mask_name), mask, compression='zlib')
    # except:
    #     pass


if __name__ == '__main__':
    import sys

    main()
    sys.exit()
