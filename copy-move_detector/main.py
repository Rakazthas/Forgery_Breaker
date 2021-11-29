# SIFT/DBSCAN detection class
import cv2.cv2 as cv2
import numpy as np
from sklearn.cluster import DBSCAN

class SIFT(object):
    def __init__(self, filepath):
        self.img = cv2.imread(filepath)

    def siftFind(self):
        sift =cv2.xfeatures2d.SIFT_create()
        grayScaleImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.keyPts, self.descriptors = sift.detectAndCompute(gray, none)
