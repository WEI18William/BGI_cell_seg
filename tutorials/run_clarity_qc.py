import os
import cv2
import pandas as pd
import numpy as np
from time import time

from cellbin.modules.iqc.clarity_qc import ClarityQC
from cellbin.utils import clog
from cellbin.image import Image


def image_split(image, name):
    _name = os.path.splitext(os.path.basename(name))[0]
    images_list = list()
    mix_value = np.max(image) + 1
    for value in range(mix_value):
        _image = (image == value).astype(np.uint8) * 255
        images_list.append(_image)
        new_name = name.replace(_name, f'{_name}_{value}')
        cv2.imwrite(new_name, _image)
    return images_list


class TestClarity(object):
    def __init__(self):
        self.cqc = ClarityQC()

    def load_model(self, model_path):
        self.cqc.load_model(model_path)

    def run(self, img_path, save_dir, name='unknown'):
        ir = Image()
        ir.read(img_path)
        img = ir.image
        img = np.squeeze(img)  # 防止redundant axis
        # 进行图像质量分析
        self.cqc.run(img)
        # 将分析结果展示在原图上
        self.cqc.post_process()
        # 可保存图片
        draw_img = self.cqc.draw_img
        print(save_dir)
        cv2.imwrite(os.path.join(save_dir, f"{name}_clarity_qc.tif"), draw_img)  # optional
        self.cqc.cluster()
        self.cqc.fig.savefig(os.path.join(save_dir, f"{name}_clarity_qc.png"), )

    def batch_test(self, img_dir, save_dir):
        results = []
        for i in os.listdir(img_dir):
            clog.info(f"----------{i}")
            t1 = time()
            sn = i.split(".")[0]
            img_path = os.path.join(img_dir, i)
            if not os.path.isdir(os.path.join(save_dir, sn)):
                os.makedirs(os.path.join(save_dir, sn))
            _save_dir = os.path.join(save_dir, sn)
            self.run(img_path, _save_dir, sn)
            t2 = time()
            clog.info(f"{i}_{t2 - t1}")
            img_preds = np.array(self.cqc.preds[:, :, 0], dtype=np.uint8)
            _ = image_split(img_preds, os.path.join(_save_dir, f"{sn}_preds.tif"))
            cv2.imwrite(os.path.join(_save_dir, f"{sn}_preds.tif"), img_preds)
            results.append([sn, self.cqc.score, t2 - t1])
        df = pd.DataFrame(results, columns=['name', 'score', 'time'])
        df_save_path = os.path.join(save_dir, "clarity_result.csv")
        df.to_csv(df_save_path, index=False)


def args_parse():
    VERSION = '1.0.0'
    usage = 'python run_clarity_qc.py -i img_dir -o save_dir -w weight_path'
    import argparse
    ArgParser = argparse.ArgumentParser(usage=usage)
    ArgParser.add_argument("--version", action="version", version=VERSION)
    ArgParser.add_argument("-i", "--input", action="store", dest="input_path", type=str, required=True,
                           help="Image folder")
    ArgParser.add_argument("-o", "--output", action="store", dest="output_path", type=str, required=True,
                           help="Result save directory")
    ArgParser.add_argument("-w", "--weight", action="store", dest="weight_path", type=str, required=True,
                           help="Clarity model weight path")
    (para, args) = ArgParser.parse_known_args()
    return para, args


def main(img_dir, save_dir, model_path):
    test_clarity = TestClarity()
    test_clarity.load_model(model_path)
    os.makedirs(save_dir, exist_ok=True)
    test_clarity.batch_test(img_dir, save_dir=save_dir)


if __name__ == '__main__':
    # img_dir = r"C:\Users\test\Desktop\20230606_test\tmp_test\test_imgs"
    # save_dir = r"C:\Users\test\Desktop\20230606_test\tmp_test\test_results"
    # model_path = r"D:\PycharmProjects\timm\output\train\20230720-172158-mobilenetv3_small_050-64\mobilenetv3_small_050.onnx"
    para, args = args_parse()
    main(
        img_dir=para.input_path,
        save_dir=para.output_path,
        model_path=para.weight_path,
    )
