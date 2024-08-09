import os

from cellbin.modules.calibrate import MifCalibrate

if __name__ == "__main__":
    dapi_img_path = r"D:\02.data\mif\C01834A5_fov_stitched.tif"
    if_img_path = r"D:\02.data\mif\C01834A5_ATP_fov_stitched.tif"

    ca = MifCalibrate(dapi_img=dapi_img_path, if_img=if_img_path)
    if_img, result = ca.calibration()
    print(1)

