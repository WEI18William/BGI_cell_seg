import glob
import os
import cv2

from cellbin.modules.iqc.track_pt import TrackPointQC
from cellbin.image.augmentation import pt_enhance_method
from cellbin.modules.iqc.track_pt import no_enhance
from cellbin.image import Image
from cellbin.image.wsi_split import SplitWSI
from cellbin.dnn.weights import auto_download_weights


def pts_on_img(img, pts, radius=1, color=(0, 255, 255), thickness=3):
    for pt in pts:
        pos = (int(pt[0]), int(pt[1]))
        cv2.circle(img, pos, radius, color, thickness)
    return img


class TestTrackPt(object):
    def __init__(self):
        self.tp_qc = TrackPointQC()
        self.img_dict = {}

    def load_model(self, p):
        self.tp_qc.ci.load_model(p)

    def set_up(self, f, p=5):
        self.tp_qc.set_enhance_func(f)
        self.tp_qc.set_multi_process(p)

    def run_detect(self, img_dir: str):
        img_dict = {f"{name}": os.path.join(img_dir, name) for name in
                    os.listdir(img_dir)}
        self.img_dict = img_dict
        self.tp_qc.track_detect(
            img_dict=img_dict
        )

    def large_split(self, large_path, save_dir=None, h=2000, w=2000):
        image_reader = Image()
        img_path = os.path.join(large_path)
        image_reader.read(img_path)
        mosaic = image_reader.image

        wsi = SplitWSI(img=mosaic, win_shape=(h, w),
                       overlap=0, batch_size=1, need_fun_ret=False, need_combine_ret=False)
        _box_lst, _fun_ret, _dst = wsi.f_split2run()
        # if save_dir is not None
        for y_begin, y_end, x_begin, x_end in _box_lst:
            fov_ = mosaic[y_begin: y_end, x_begin: x_end]
            out_p = os.path.join(save_dir, f"{y_begin}_{y_end}_{x_begin}_{x_end}.tif")
            image_reader.write_s(image=fov_, output_path=out_p, compression=True)

    def show_track(self, save_dir, enhance_func, detect_channel=-1):
        for row_col, item in self.tp_qc.track_result.items():
            cps, angle = item
            img_obj = Image()
            img_path = self.img_dict[row_col]
            _, image_name = os.path.split(img_path)
            image_name, ext = os.path.splitext(image_name)
            img_obj.read(img_path)
            img_obj.get_channel(ch=detect_channel)
            enhance_arr = enhance_func(img_obj)

            result_arr = pts_on_img(enhance_arr, cps, radius=20)
            save_path = os.path.join(save_dir, image_name + '.tif')
            img_obj.write_s(result_arr, save_path)


def main():
    # run track detect test
    weight_path = r"C:\Users\test\Downloads\points_detect_yolov5obb_DAPI_20230309_torch.onnx"
    img_dir = r"C:\Users\test\Desktop\demo\SS200000135TL_D1_stitched_down_5.tif"
    save_dir = r"C:\Users\test\Desktop\demo\down5"
    show_result = r"C:\Users\test\Desktop\demo\show_result"
    os.makedirs(save_dir, exist_ok=True)
    enhance_func = pt_enhance_method['DAPI']
    process = 0
    track_detect = TestTrackPt()
    if not os.path.exists(weight_path):
        zoo_dir, weight_names = os.path.split(weight_path)
        auto_download_weights(zoo_dir, weight_names)
    track_detect.load_model(weight_path)

    track_detect.large_split(img_dir, save_dir, 400, 400)
    track_detect.set_up(enhance_func, process)
    track_detect.run_detect(img_dir=save_dir)
    track_detect.show_track(save_dir=show_result, enhance_func=enhance_func)
    # result, img_dict = test_track_detect(weight_path, img_dir, )

    # visualize
    # show_track(result, img_dict, save_dir, pt_enhance_method['DAPI'])

    # img_dir = r"D:\Data\tmp\Y00035MD\enhance_1"
    # save_dir = r"D:\Data\tmp\Y00035MD\enhance_1_result"
    # os.makedirs(save_dir, exist_ok=True)
    # result, img_dict = test_track_detect(weight_path, img_dir, no_enhance)
    #
    # # visualize
    # show_track(result, img_dict, save_dir, no_enhance)


if __name__ == '__main__':
    main()
