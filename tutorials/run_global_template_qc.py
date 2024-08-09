import os
import numpy as np
from cellbin.contrib.template_reference import TemplateReference

def test_template_reference(pts, index, loc, chipno, scalex, scaley, rotate):
    tr = TemplateReference()
    tr.set_scale(scalex, scaley)
    tr.set_rotate(rotate)
    tr.set_chipno(chipno)
    tr.set_fov_location(loc)
    tr.set_qc_points(index, pts)

    tr.reference_template(mode='multi')
    dct = tr.get_template_eval()
    mat = tr.get_global_eval()

    return dct, mat


if __name__ == '__main__':
    import h5py

    ipr_path = r"D:\AllData\temp_data\D01862B1_20230505_115103_0.1.ipr"
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
        scalex = conf['Register'].attrs['ScaleX']
        scaley = conf['Register'].attrs['ScaleY']
        rotate = conf['Register'].attrs['Rotation']
        # loc = conf['Stitch/BGIStitch/StitchedGlobalLoc'][...]
        loc = conf['Research/Stitch/StitchFovLocation'][...]
        index = conf['Stitch'].attrs['TemplateSource']

    chipno = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
              [240, 300, 330, 390, 390, 330, 300, 240, 420]]

    # chipno = [[112, 144, 208, 224, 224, 208, 144, 112, 160],
    #           [112, 144, 208, 224, 224, 208, 144, 112, 160]]

    dct, mat = test_template_reference(pts, index, loc, chipno, scalex, scaley, rotate)
    print(1)