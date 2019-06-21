# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:42:55 2019

@author: mjensen1
"""

import os
import sys
import pytesseract
import cv2
import numpy as np

import page_dewarp as dwp

# Set page margins for dewarping to 0 since I take care of these myself
dwp.PAGE_MARGIN_X = 0
dwp.PAGE_MARGIN_Y = 0

# path to tesseract executable
# You'll need to edit this to path to point to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\mjensen1\AppData\Local\Tesseract-OCR\tesseract.exe'


def import_img(filename, landscape=True):
    img = cv2.imread(filename)
    rows, cols = img.shape[:2]
    if landscape and rows > cols:
        return np.rot90(img, axes=(0,1))
    else:
        return img


def view_img(img, window_name='image'):
    cv2.imshow(window_name, img);
    cv2.waitKey(0);
    cv2.destroyAllWindows()


def shrink(img, new_height=500):
    ratio = new_height/img.shape[0]
    return ratio, cv2.resize(img, None, fx=ratio, fy=ratio,
                             interpolation=cv2.INTER_AREA)


def get_edges(img, lower=50, upper=100, normalize=True):
    # Apply canny filter to find edges
    edged = cv2.Canny(img, lower, upper)
    # Median blur gets rid of thin edges, which are probably not next
    edged = cv2.medianBlur(edged, 3)
    # Dilate remaining pixels so we don't "jump" over them later
    edged = cv2.dilate(edged, np.ones((2,2)), iterations=1)
    if normalize:
        edged = edged / edged.max()
    return edged


def find_vertical_bounds(edged_img):
    '''Finds the left/right edges of the blocks of text on both pages in the
    image. Takes an image that has already been passed through shrink() and
    get_edges(). Should also already be normalized.
    '''
    # Add all rows together
    col_sums = np.sum(edged_img, axis=0)
    width = len(col_sums)
    # Find left edge of left sheet
    left_edge1 = 0
    for i in range(width//2):
        if np.min(col_sums[i:i+20]) > 5:
            left_edge1 = i
            break
    # Find right edge of left sheet
    right_edge1 = 0
    for i in range(left_edge1, 3*width//4):
        if np.max(col_sums[i+1:i+11]) < 5:
            right_edge1 = i
            break
    # Find left edge of right sheet
    left_edge2 = 0
    for i in range(right_edge1, width):
        if np.min(col_sums[i:i+25]) > 5:
            left_edge2 = i
            break
    # Find right edge of right sheet
    right_edge2 = 0
    for i in range(left_edge2, width):
        if np.max(col_sums[i+1:i+11]) < 5:
            right_edge2 = i
            break
    return left_edge1, right_edge1, left_edge2, right_edge2


def get_pages(img, margin=10):
    '''Takes an image of an open book and cuts the two individual pages out.
    '''
    # Shrink things down so it's less computationally expensive to find borders
    ratio, shrunk = shrink(img)
    edged_img = get_edges(shrunk) # Creates binary map of image
    # Determine where the left/right edges of text blocks are
    left1, right1, left2, right2 = find_vertical_bounds(edged_img)
    # Undo shrinking and add margins
    left1 = int((left1 - margin)/ratio)
    right1 = int((right1 + margin)/ratio)
    left2 = int((left2 - margin)/ratio)
    right2 = int((right2 + margin)/ratio)
    # Slice pages out of original image
    page1 = img[:,left1:right1]
    page2 = img[:,left2:right2]
    return page1, page2


def flatten_page(page, name):
    '''Uses functions in page_dewarp.py to flatten text on image of a page
    Basically just copies the functionality of the main() method in page_dewarp
    '''
    small = dwp.resize_to_screen(page)
    pagemask, page_outline = dwp.get_page_extents(small)
    cinfo_list = dwp.get_contours(name, small, pagemask, 'text')
    spans = dwp.assemble_spans(name, small, pagemask, cinfo_list)
    
    if len(spans) < 3:
        print('Detecting lines because only {} text spans'.format(len(spans)))
        cinfo_list = dwp.get_contours(name, small, pagemask, 'line')
        spans2 = dwp.assemble_spans(name, small, pagemask, cinfo_list)
        if len(spans2) > len(spans):
            spans = spans2
    if len(spans) < 1:
        print('Only {} spans in {}. Returning original image.'.format(name, len(spans)))
        return page
    span_points = dwp.sample_spans(small.shape, spans)
    
    corners, ycoords, xcoords = dwp.keypoints_from_samples(name, small,
                                                           pagemask,
                                                           page_outline,
                                                           span_points)
    rough_dims, span_counts, params = dwp.get_default_params(corners, ycoords,
                                                             xcoords)
    dstpoints = np.vstack((corners[0].reshape((1,1,2)),) + tuple(span_points))
    params = dwp.optimize_params(name, small, dstpoints, span_counts, params)
    page_dims = dwp.get_page_dims(corners, rough_dims, params)
    flattened = dwp.remap_image(name, page, small, page_dims, params, True)
    return flattened


def process_img(filename):
    img = import_img(filename)
    page1, page2 = get_pages(img)
    print('Flattening left page...')
    try:
        page1_flat = flatten_page(page1, 'page1')
        #view_img(shrink(page1_flat, 1000)[1])
        print('Running OCR on left page...')
        page1_text = pytesseract.image_to_string(page1_flat)
    except cv2.error:
        print('OCR failed on {}, left page.'.format(filename))
        page1_text = '*NO OCR FOR THIS PAGE*'
    print('Flattening right page...')
    try:
        page2_flat = flatten_page(page2, 'page2')
        #view_img(shrink(page2_flat, 1000)[1])
        print('Running OCR on right page...')
        page2_text = pytesseract.image_to_string(page2_flat)
    except cv2.error:
        print('OCR failed on {}, right page.'.format(filename))
        page2_text = '*NO OCR FOR THIS PAGE*'
    return page1_text, page2_text


def read_book_from_folder(folder_path, saveas=None):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) \
             if f[-4:].lower() == '.jpg']
    pages = []
    for f in files:
        print('Processing {}...'.format(f))
        page1, page2 = process_img(f)
        pages += [page1, page2]
        if saveas is not None:
            string_out = page1 + '\n\n' + page2 + '\n\n'
            with open(saveas, 'a', encoding='utf-8') as fh:
                fh.write(string_out)
        print()
    return pages


if __name__ == '__main__':
    if len(sys.argv) == 3:
        read_book_from_folder(sys.argv[1], sys.argv[2])
    else:
        print('Syntax is "python ocr.py folder_path saveas"')