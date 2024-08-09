from cellbin.modules.stitching import Stitching
from tutorials.run_stitch import imagespath2dict

def test_stitch_qc(image_dict, fov_location=None, rows=None, cols=None):
    """
    用于显微镜拼接坐标评估
    Args:
        image_dict:
        fov_location:
    Return:

    """
    if fov_location is not None:
        rows, cols = fov_location.shape[:2]
    elif rows is None or cols is None:
        return

    stitch = Stitching()
    stitch.set_size(rows, cols)
    # stitch._init_parm(image_dict)
    # stitch._get_jitter(image_dict)
    stitch.stitch(image_dict)
    scope_x_dif, scope_y_dif = stitch._get_jitter_eval()
    cds_x_dif, cds_y_dif = stitch._get_stitch_eval()

    return [scope_x_dif, scope_y_dif], [cds_x_dif, cds_y_dif]

if __name__ == "__main__":
    image_path = r'D:\AllData\temp_data\zh\dxc\23\Images'
    test_stitch_qc(imagespath2dict(image_path), rows=15, cols=16)

