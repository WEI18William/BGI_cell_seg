# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:49:06 2023

@author: wuyiwen
"""

import os
import cv2
import numpy as np
import tifffile as tifi
import pandas as pd
from cellbin.utils import clog
from cellbin.image import Image
from stio.matrix_loader import MatrixLoader  # input is file path
from cellbin.modules.cell_labelling import CellLabelling


class LabelingInterface(object):
    def __init__(self, mask_path, gem_path, output_path='./'):
        self.mask = None
        self.mask_path = mask_path
        self.gem = None
        self._gem_path = gem_path
        self.file_name = '111'
        self.bgef = None
        self.exp_matrix = None
        self._output_path = output_path

    def run(self):
        # try:
        mask = tifi.imread(self.mask_path)
        if self._gem_path.lower().endswith(('.bgef', '.gef')):
            new_file = os.path.join(self._output_path, f"{self.file_name}_exp.gem")
            from gefpy.bgef_reader_cy import BgefR
            self.bgef = BgefR(self._gem_path, 1, 8)
            ml = MatrixLoader(self._gem_path)
            ml.bgef2gem(bgef_file=self._gem_path, gem_file=new_file, binsize=1)
        else:
            new_file = self._gem_path
        cl = CellLabelling(
            mask,
            new_file
        )
        cl.set_process(6)
        correct_mask, self.exp_matrix = cl.run_fast()
        gem_path = os.path.join(self._output_path, f"{self.file_name}_exp_correct.gem")
        self.exp_matrix.to_csv(gem_path, sep='\t', index=False)
        ml = MatrixLoader(gem_path)
        bef_path = os.path.join(self._output_path, f"{self.file_name}_exp_correct.bgef")
        # ml.gem2bgef(
        #     gem_file=gem_path,
        #     bgef_file=bef_path,
        #     binsize=[1],
        #     region=None
        # )
        correct_mask = cl.draw_corrected_mask(correct_mask)
        correct_mask_path = os.path.join(self._output_path, f'{self.file_name}_cell_mask_correct.tif')
        Image.write_s(correct_mask, correct_mask_path, compression=True)

        cgef_path = os.path.join(self._output_path, f"{self.file_name}_exp_correct.cgef")
        ml.cgem_to_cgef(
            cgem_file=gem_path,
            outpath=cgef_path,
            block_size=[256, 256]
        )
        clog('Done!')

    # except Exception as e:
    #     clog(e)


if __name__ == '__main__':
    # a = MatrixLoader(r"D:\git_libs\cellbin\test\C01333A3.gem.gz")
    # a.gem2bgef(gem_file = r"D:\git_libs\cellbin\test\C01333A3.gem.gz", bgef_file=r"D:\git_libs\cellbin\test\C01333A3.bgef", binsize=[1],region=None)
    # a.bgef2gem(bgef_file=r"D:\git_libs\cellbin\test\C01333A3.bgef", gem_file = r"D:\git_libs\cellbin\test\C01333A3_test.gem.gz", binsize=1)
    # from gefpy.bgef_reader_cy import BgefR
    # bgef_reader = BgefR(r"D:\git_libs\cellbin\test\C01333A3.bgef",1,8)
    # explist = bgef_reader.get_expression()
    # gene_data = bgef_reader.get_gene_data()
    # clog(explist)
    # print(gene_data)
    # print(len(gene_data[1]))

    iface = LabelingInterface(mask_path=r"D:\yiwen_correct_data\FP200000443TL_F4_5000_5000_mask.tif",
                              gem_path=r"D:\yiwen_correct_data\FP200000443TL_F4_5000_5000.gem",
                              output_path=r"D:\yiwen_correct_data"
                              )
    iface.run()
