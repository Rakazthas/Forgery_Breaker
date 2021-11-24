import itertools

import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct

from math import copysign,log10

from skimage import data
from skimage.measure import shannon_entropy
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

blockSize = 8
logRadius = blockSize/2

imgName = input("Entrer le chemin de l'image : ")
img = cv2.imread(imgName)
h,w = img.shape[:2]

imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
imgY = cv2.split(imgYCC)[0]

blocks = {}

'''
*************************************************
creation blocs
*************************************************
'''

for i in range (0, h - blockSize, int(blockSize/2)):
    for j in range (0, w - blockSize, int(blockSize/2)):
        array = imgY[i:i + (blockSize), j:j + (blockSize)]
        if shannon_entropy(array) > 4.5:
            blocks[i, j] = array

print("Nombre de blocks : {}".format(len(blocks)))


'''
*************************************************
transformation en coordonnees polaires
*************************************************
'''
BlockLog = {}

for key, value in blocks.items():
    value = value.astype(np.float32)
    root = np.sqrt(((value.shape[0]/2.0)**2.0) + ((value.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(value, (value.shape[0]/2, value.shape[1]/2), blockSize/2, cv2.WARP_POLAR_LOG)
    polar_image - polar_image.astype(np.uint8)
    BlockLog[key] = polar_image

coordsDup = []
listMean = {}

for x, y in itertools.combinations(BlockLog.keys(), 2):
    key1 = x
    key2 = y
    val1 = BlockLog.get(key1)
    val2 = BlockLog.get(key2)

    G_a = np.fft.fft2(val1)
    G_b = np.fft.fft2(val2)

    Conj_B = np.ma.conjugate(G_b)

    R = G_a * Conj_B
    R /= np.absolute(R)
    r = np.fft.ifft2(R)
    nr = np.absolute(r)

    _mean = np.mean(nr)
    listMean[key1, key2] = _mean
    print("Mean du bloc [{}, {}] : {}".format(key1, key2, _mean))



'''
imgRe = np.zeros_like(imgY)
for key, value in blocks.items():
    i,j = key[:2]
    imgRe[i:i+(blockSize), j:j + (blockSize)] = value

cv2.cv2.imshow('retrait fond',imgRe)
'''
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
