# Log-Polar detection script
# Base imports
import cv2.cv2
import matplotlib.pyplot as plt
import numpy
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

'''
print("Type de f : {}".format(type(f)))
print("Type de f[0] : {}".format(type(f[0])))
print("Val de f[0] : {}".format(f[0]))
'''
nbBlocks = len(blocksLog)
L = int(nbBlocks*(nbBlocks-1)/2)
listEquals = []
listMax = {}
listMean = {}

for i in range(0, nbBlocks):
    for j in range(i+1, nbBlocks):
        G_a = np.fft.fft2(blocksLog[i])
        G_b = np.fft.fft2(blocksLog[j])

        conj_b = np.ma.conjugate(G_b)

        R = G_a*conj_b
        R /= np.absolute(R)
        r = np.fft.ifft2(R)
        nr = np.absolute(r)

        _max = np.amax(nr)
        _mean = np.mean(nr)

        listMax[i, j] = _max
        listMean[i, j] = _mean

        print("Max du bloc [{}, {}] : {}".format(i, j, _max))

'''
for i in range(0, len(blocksFFT)):
    for j in range(i+1, len(blocksFFT)):

        g = numpy.empty(shape=len(blocksFFT[i]))
        for x in range (0, len(blocksFFT[i])):
            h = numpy.empty(shape=len(blocksFFT[i][x]))
            for y in range (0, len(blocksFFT[i][x])):
                val = blocksFFT[i][x][y] * blocksFFT[j][x][y].conjugate()
                if val != 0:
                    val = val/abs(val)
                np.append(h, val)
            np.append(g, h)

        _max = []
        _mean = []

        invG = np.fft.ifft2(g)

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
'''

'''
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

        if finMax > 0.35:
            listEquals.append([i, j])
'''

globMeanList = list(listMean.values())
globMean = np.mean(globMeanList)

#globMean = 0

'''
for i in range(0, nbBlocks):
    for j in range(i+1, nbBlocks):
        globMean += listMean[i, j]

_mean = globMean/L
'''


#globMean = np.mean(listMean)
'''
for i in range(0, len(blocksLog)):
    for j in range(i+1, len(blocksLog)):
        if listMax[i, j] > 0.35: #globMean:
            listEquals.append([i, j])
'''

for key, value in listMean.items():
    print("key : {}, value : {}".format(key, value))
    if value > 0.35:
        listEquals.append(key)

#print("Moyennes : {}".format(listMean))
print("Moyenne : {}".format(globMean))
#TODO : check why diff max/listMax.value


print("Liste de couples : {}".format(listEquals))

