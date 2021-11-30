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

    def siftShowFeatures(self):
        siftImg = cv2.drawKeypoints(self.img, self.keyPts, self.img.copy())
        return siftImg

    def forgeryLocate(self, eps=40, minSamples=2):
        clusters = DBSCAN(eps=eps, min_samples=minSamples).fit(self.descriptors)
        size = np.unique(clusters.labels_).shape[0] - 1
        forgery = slef.img.copy()
        if size == 0 & np.unique(clusters.labels_)[0] == -1:
            print("No forgery found")
            return None
        else:
            if size == 0:
                size=1
        clustersList=[[] for i in range(size)]
        for i in range(len(self.keyPts)):
            if clusters.labels_[i] != -1:
                clustersList[clusters.labels_[i]].append((int(self.keyPts[i].pt[0]), int(self.keyPts[i].pt[1])))
        for pts in clustersList:
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(forgery, pts[0], pts[i], (0,0,255), 2)
        return forgery