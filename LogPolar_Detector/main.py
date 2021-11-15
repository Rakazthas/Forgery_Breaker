# Log-Polar detection script
# Base imports
import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
from math import copysign,log10
import scipy
import statistics

# Log-Polar operations
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

blockSize = 64
logRadius = blockSize/2

imgName = input("Entrer le chemin de l'image : ")
img = cv2.imread(imgName)
h,w = img.shape[:2]


print("Largeur : {} px,hauteur:{} px".format(w,h))
px_value =img[20,100]
print("Valeur du pixel en (20,100): {}".format(px_value))
cv2.imshow('img', img)

#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()

imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
imgY = cv2.cv2.split(imgYCC)[0]

blockPerLine = int(w/blockSize)
blocks = []

for i in range (0, h - blockSize, blockSize):
    for j in range (0, w - blockSize, blockSize):
        array = imgY[i:i+blockSize, j:j+blockSize]
        blocks.append(array)

#cv2.imshow("block i", blocks[5 + 5*blockPerLine])
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()

blocksLog = []

for x in blocks:
    imgFl = x.astype(np.float32)
    value = np.sqrt(((imgFl.shape[0]/2.0)**2.0 + (imgFl.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(imgFl, (imgFl.shape[0]/2, imgFl.shape[1]/2), blockSize/2, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    blocksLog.append(polar_image)

#cv2.imshow("block log", blocksLog[5 + 5*blockPerLine])
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


blocksFFT = []

for x in blocksLog:
    f = np.fft.fft2(x)
    blocksFFT.append(f)

listEquals = []

listMax = {}
listMean = {}
for i in range(0, len(blocksLog)):
    for j in range(i+1, len(blocksLog)):
        f, cross = scipy.signal.csd(blocksLog[i], blocksLog[j], scaling = 'spectrum')

        _norm = np.linalg.norm(cross)
        nCross = cross/_norm

        _max = []
        _mean = []

        invG = np.fft.ifft2(nCross)

        absInvG = []

        for x in invG:
            absI = []
            for y in x:
                absI.append(abs(y))
            absInvG.append(absI)

        for x in absInvG:
            _max.append(max(x))
            _mean.append(statistics.mean(x))
        finMax = max(_max)
        finMean = statistics.mean(_mean)

        listMax[i, j] = finMax
        listMean[i, j] = finMean

        '''if finMax > 0.35:
            listEquals.append([i, j])'''

_mean = 0
L = len(blocksLog)*(len(blocksLog)-1)/2
for i in range(0, len(blocksLog)):
    for j in range(i+1, len(blocksLog)):
        _mean += listMean[i, j]

_mean = _mean/L
'''
for i in range(0, len(blocksLog)):
    for j in range(i+1, len(blocksLog)):
        if listMax[i, j] > _mean:
            listEquals.append([i, j])'''

for i in range(0, len(blocksLog)):
    for j in range(i+1, len(blocksLog)):
        print("Max du bloc [{}, {}] : {}".format(i, j, listMax[i, j]))

print("Moyenne : {}".format(_mean))
#TODO : correlation test error


print("Liste de couples : {}".format(listEquals))

