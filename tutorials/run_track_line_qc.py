import os
vipshome = r'C:\vips-dev-8.12\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

from cellbin.modules.iqc.track_line import TrackLineQC
from cellbin.modules import StainType
from cellbin.image.augmentation import line_enhance_method


def track_line_detect(src_dir, save_dir=None, angle=None):
    img_dict = {name: [os.path.join(src_dir, name), angle] for name in os.listdir(src_dir)}
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    track_line_qc = TrackLineQC(magnification=10, scale_range=0.8)
    track_line_qc.set_preprocess_func(line_enhance_method.get(StainType.HE.value, None))
    track_line_qc.set_debug_mode(True, save_dir)
    track_line_qc.line_detect(img_dict)
    track_line_qc.track_match()


if __name__ == '__main__':
    src_dir = r"D:\Data\new_qc\HE_test\A01012B1\A01012B1\N01012B"
    save_dir = r"D:\Data\new_qc\HE_test\A01012B1\track_line_out"
    track_line_detect(src_dir=src_dir, save_dir=save_dir)
