# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 00:02:38 2017

@author: Shao Yn
"""

from scipy.misc import imread


def main():
    #content_image为景物图片
    content_image = read(path_style)
    #style_image为艺术风格图片
    style_image = read(path_style)
    
    initial_image = content_image
    
    convert(content_image, style_image)
    
    

def read(path):
    img = read(path_style).astype(np.float)
    return img
    