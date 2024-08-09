from cellbin.image.wsi_split import SplitWSI
from cellbin.utils import clog
from cellbin.dnn.cseg import CellSegmentation
from cellbin.dnn.cseg.predict import CellPredict
from cellbin.dnn.cseg.processing import f_prepocess, f_preformat, f_postformat, f_preformat_mesmer, \
    f_postformat_mesmer, f_padding, f_fusion, f_postprocess_v2, f_preformat_rna, f_postformat_rna, f_postprocess_rna
from cellbin.dnn.onnx_net import OnnxNet
from cellbin.modules import StainType
import pdb
from skimage.morphology import remove_small_objects
import numpy as np
from typing import Optional


# TensorRT/ONNX
# HE/DAPI/mIF
class Segmentation(CellSegmentation):

    def __init__(
            self,
            model_path="",
            net="bcdu",
            mode="onnx",
            gpu="-1",
            num_threads=0,
            win_size=(256, 256),
            intput_size=(256, 256, 3),
            overlap=16,
            img_type=''
    ):
        """

        :param model_path:
        :param net:
        :param mode:
        :param gpu:
        :param num_threads:
        """
        # self.PREPROCESS_SIZE = (8192, 8192)

        self._win_size = win_size
        self._input_size = intput_size
        self._overlap = overlap
        self.watershed_win_size = (4900,4900)

        self._net = net
        self._gpu = gpu
        self._mode = mode
        # self._model_path = model_path
        self._model: Optional[OnnxNet] = None
        self._sess: Optional[CellPredict] = None
        self._num_threads = num_threads
        self.img_type = img_type
        # self._f_init_model()

    def f_init_model(self, model_path):
        """
        init model
        """
        self._model = OnnxNet(model_path, self._gpu, self._num_threads)
        if self.img_type.upper() == StainType.rna.value:
            self._sess = CellPredict(self._model, f_preformat_rna, f_postformat_rna)
        else:
            if self._net == "mesmer":
                self._sess = CellPredict(self._model, f_preformat_mesmer, f_postformat_mesmer)
            else:
                self._sess = CellPredict(self._model, f_preformat, f_postformat)

    def f_predict(self, img):

        """

        :param img:CHANGE
        :return: 掩模大图
        2023/09/21 @fxzhao 设置need_fun_ret为False,当前版本未用到此结果
        """
        img = f_prepocess(
            img=img,
            img_type=self.img_type
        )
        sp_run = SplitWSI(
            img=img,
            win_shape=self._win_size,
            overlap=self._overlap,
            batch_size=100,
            need_fun_ret=False,
            need_combine_ret=True,
            editable=False,
            tar_dtype=np.uint8,
            dst_shape=(img.shape[:2]),
            win_back=True
        )
        sp_run.f_set_run_fun(self._sess.f_predict)
        sp_run.f_set_pre_fun(f_padding, self._win_size)
        # sp_run.f_set_fusion_fun(f_fusion)
        _, _, pred_raw = sp_run.f_split2run()
        
        if self.img_type.upper() == StainType.rna.value:
            pred = f_postprocess_rna(pred_raw)
        else:
            #post processing
            sp_run2 = SplitWSI(pred_raw, self.watershed_win_size, self._overlap, 1, False, True, False, np.uint8)
            sp_run2.f_set_run_fun(f_postprocess_v2)
            sp_run2.f_set_pre_fun(f_padding, self.watershed_win_size)
            sp_run2.f_set_fusion_fun(f_fusion)
            _, _, pred = sp_run2.f_split2run()
            pred = remove_small_objects(pred.astype(np.bool8), min_size=15, connectivity=2).astype(np.uint8)
        return pred


def main():
    import tifffile
    import os
    import argparse

    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-g", "--gpu", help="the gpu index", default=1)
    parser.add_argument("-n", "--net", help="bcdu or mesmer", default="bcdu")
    parser.add_argument("-m", "--mode", help="onnx or tf", default="onnx")
    parser.add_argument("-th", "--num_threads", help="num_threads", default="0")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    gpu = args.gpu
    mode = args.mode
    num_threads = args.num_threads
    net = args.net
    if input_path is None or output_path is None:
        print("please check your parameters")
        sys.exit()
    print(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    abs_path = os.path.dirname(os.path.abspath(__file__))
    if mode == "onnx":
        if net == "mesmer":
            model_path = os.path.join(abs_path, r"D:\code\public\cell_segmentation_v03\model/weight_mesmer_20.onnx")
        else:
            model_path = os.path.join(abs_path, r"D:\code\public\cell_segmentation_v03\model/weight_cell_221008.onnx")
    else:
        model_path = os.path.join(abs_path, r"D:\code\public\cell_segmentation_v03\model/weight_cell_221008.hdf5")
        # model_path = os.path.join(abs_path, "model/weight_cell_he_20221226.hdf5")
    img = tifffile.imread(input_path)
    clog.info(f"start loading model from {model_path}")
    sg = Segmentation(model_path=model_path, net=net, mode=mode, gpu=gpu, num_threads=int(num_threads))
    sg.f_init_model(model_path=model_path)
    clog.info(f"model loaded,start prediction")
    pred = sg.f_predict(img)
    clog.info(f"prediction finish,start writing")
    tifffile.imwrite(output_path, pred)
    clog.info(f"finish!")


if __name__ == '__main__':
    import sys

    main()
    sys.exit()
