import numpy as np
from glob import glob
import os
import cv2
import tifffile
import h5py
from shutil import copyfile
import multiprocessing as mp

from cellbin.modules.registration import Registration
from cellbin.utils import clog
from stio.chip import STOmicsChip


def merge3ch(ch0, ch1):
    h0, w0 = ch1.shape
    hr, wr = (int(h0 / 3), int(w0 / 3))
    dim = (wr, hr)
    ch1r = cv2.resize(ch1, dim)
    ch0r = cv2.resize(ch0, dim)
    arr = np.zeros((hr, wr, 3), dtype=np.uint8)
    arr[:, :, 0] = ch0r
    arr[:, :, 2] = 100 * ch1r

    return arr


def ipr_to_regist(regist_path, output_dir):
    chip_template_getter = STOmicsChip()
    os.makedirs(output_dir, exist_ok=True)

    files = list(filter(os.path.isfile, glob(regist_path + "/*.ipr")))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    the_most_recent_ipr = files[0]

    # old_ipr_path = glob(os.path.join(regist_path, "**.ipr"))[0]
    old_ipr_path = the_most_recent_ipr
    ipr_name = os.path.split(old_ipr_path)[-1]
    new_ipr_path = os.path.join(output_dir, ipr_name)
    clog.info(f"Copying from {old_ipr_path} to {new_ipr_path}")
    copyfile(old_ipr_path, new_ipr_path)
    with h5py.File(new_ipr_path, "r") as f:
        # json_obj = json.load(f)
        chip_name = f['ImageInfo'].attrs['STOmicsChipSN']
        regist_module = f["Register"]
        scale_x = regist_module.attrs["ScaleX"]
        scale_y = regist_module.attrs["ScaleY"]
        rotation = regist_module.attrs["Rotation"]
        old_rot = regist_module.attrs['CounterRot90']
        old_offset = [regist_module.attrs['OffsetX'], regist_module.attrs['OffsetY']]

    new_regist_save_path = os.path.join(output_dir, f"{chip_name}_regist_new.tif")
    # if os.path.exists(new_regist_save_path):
    #     clog.info(f"Result exists, skip {chip_name}")
    #     return
    short_sn = chip_template_getter.get_valid_chip_no(chip_name)
    stereo_chip = chip_template_getter.get_chip_grids(short_sn)
    if len(glob(os.path.join(regist_path, '**fov_stitched.tif'))) == 0:
        clog.info(f"No result, skip {chip_name}")
        return
    fov_stitched_path = glob(os.path.join(regist_path, '**fov_stitched.tif'))[0]
    fov_stitched = tifffile.imread(fov_stitched_path)
    print(fov_stitched_path)

    # czi mouse brain -> stitch shape (2, x, x)
    if len(fov_stitched.shape) == 3:
        fov_stitched = fov_stitched[0, :, :]

    # try:
    #     gene_exp_path = glob(os.path.join(regist_path, "**raw.tif"))[0]
    # except IndexError:
    #     try:
    #         gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**_gene_exp.tif"))[0]
    #     except IndexError:
    #         gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**.gem.tif"))[0]

    gene_exp_path = glob(os.path.join(regist_path, "**gene.tif"))[0]
    gene_exp = cv2.imread(gene_exp_path, -1)
    print(gene_exp_path)
    track_template = np.loadtxt(glob(os.path.join(regist_path, '**stitch_template.txt'))[0])  # stitch template
    flip = True
    # im_shape = np.loadtxt(os.path.join(regist_path, "4_register", "im_shape.txt"))
    rg = Registration()
    rg.mass_registration_stitch(
        fov_stitched,
        gene_exp,
        stereo_chip,
        track_template,
        1 / scale_x,
        1 / scale_y,
        rotation,
        flip
    )
    clog.info(f"old-> {old_rot}, {old_offset}")
    clog.info(f"new-> {rg.offset}, {rg.rot90}, {rg.score}")
    clog.info(f"offset-> {np.array(old_offset) == np.array(rg.offset)}")
    clog.info(f"rot-> {rg.rot90 == old_rot}")
    rg.transform_to_regist()
    regist_img = rg.regist_img
    with h5py.File(new_ipr_path, 'a') as conf:
        regist_module = conf["Register"]
        regist_module.attrs['CounterRot90'] = rg.rot90
        regist_module.attrs['OffsetX'] = rg.offset[0]
        regist_module.attrs['OffsetY'] = rg.offset[1]
    tifffile.imwrite(new_regist_save_path, regist_img)
    # merge_result = merge3ch(regist_img, gene_exp)
    # tifffile.imwrite(os.path.join(output_dir, f"{chip_name}_regist_merged.tif"), merge_result)


def run_multi(src_dir, save_dir):
    for i in os.listdir(src_dir):
        clog.info(f"Working on ----------- {i}")
        pipeline_dir = os.path.join(src_dir, i, 'pipeline', 'registration')
        cur_save_dir = os.path.join(save_dir, i)
        try:
            ipr_to_regist(regist_path=pipeline_dir, output_dir=cur_save_dir)
        except:
            continue


def run_multi_multiprocess(src_dir, save_dir):
    pool = mp.Pool(processes=30)
    for i in os.listdir(src_dir):
        clog.info(f"Working on ----------- {i}")
        pipeline_dir = os.path.join(src_dir, i, 'pipeline', 'registration')
        cur_save_dir = os.path.join(save_dir, i)
        try:
            sub_process = pool.apply_async(ipr_to_regist, args=(pipeline_dir, cur_save_dir))
        except:
            continue
    pool.close()
    pool.join()


if __name__ == '__main__':
    import time
    start = time.time()
    # test_path = r"/storeData/USER/data/01.CellBin/01.data/01.StereoCell/01.dapi/02.pipeline/630testdata/result/SS200000330TR_D1/pipeline/registration/"
    # save_dir = r"/storeData/USER/data/01.CellBin/00.user/dengzhonghan/data/630_registration_rerun_20230704/SS200000330TR_D1"

    test_path = r"D:\Data\new_pipeline\regist_issue\A02084A3"
    save_dir = r"D:\Data\new_pipeline\regist_issue\new_result_dir\A02084A3"
    ipr_to_regist(test_path, save_dir)
    end = time.time()
    print(f"time cost {end - start}")

    # src_dir = "/storeData/USER/data/01.CellBin/01.data/01.StereoCell/01.dapi/02.pipeline/630testdata/result"
    # save_dir = "/storeData/USER/data/01.CellBin/00.user/dengzhonghan/data/630_registration_rerun_20230704"
    # run_multi_multiprocess(src_dir, save_dir)
