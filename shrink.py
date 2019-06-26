# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:16:58 2019

@author: mjensen1

Shrink images in a given directory
"""

import cv2
import os
import ocr

SHRUNK_HEIGHT = 750

def import_shrink_export(input_file, output_file, ccw=True):
    img = ocr.import_img(input_file, ccw=ccw)
    _, img_small = ocr.shrink(img, SHRUNK_HEIGHT)
    cv2.imwrite(output_file, img_small)

def shrink_all_in_dir(input_dir, output_dir, ccw=True):
    '''Iterates through all files in a directory, shrinks them, and writes the
    shrunk versions to another directory.
    '''
    files = os.listdir(input_dir)
    for file in files:
        print('Working on {}...'.format(file))
        try:
            import_shrink_export(os.path.join(input_dir, file),
                                 os.path.join(output_dir, file), ccw=ccw)
        except AttributeError:
            print('Skipping {}...'.format(file))