# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:18:27 2023

@author: HXF
"""
import cv2
import math
import numpy as np

def DarkChannel(im):
            b, g, r = cv2.split(im)
            dc = cv2.min(cv2.min(r, g), b);
            return dc

def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort(0)
    indices = indices[(imsz - numpx):imsz]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def DarkIcA(im, A):
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    return DarkChannel(im3)