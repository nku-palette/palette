# coding: utf-8
"""
Label image and raw data processing.
"""

import os

def write_txt(images_dir, labels, txt_name):
    """
    Write txt of image name and label.
    """
    image_name = filter(lambda x: '.JPG' in x, os.listdir(images_dir))
    path = images_dir + '\\' + str(txt_name) + '.txt'
    with open(path, 'a') as f:
        for each in image_name:
            f.writelines([each + ' ' + labels + '\n'])

def extract_box(box_dir):
    """
    Extract bounding box ground-truth from xml annotations labeled manually.
    """
    pass

if __name__ == '__main__':
    data_dir = 'D:\OD\myCat'
    write_txt(data_dir, 'cat', 'train')