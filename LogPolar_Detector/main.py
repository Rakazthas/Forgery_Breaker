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

blockSize = 8
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

'''
*************************************************
creation blocs
*************************************************
'''
for i in range (0, h, blockSize):
    for j in range (0, w, blockSize):
        array = imgY[i:i+(blockSize-1), j:j+(blockSize-1)]
        blocks.append(array)

#cv2.imshow("block i", blocks[5 + 5*blockPerLine])
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()

blocksLog = []

'''
*************************************************
transformation en coordonnees polaires
*************************************************
'''
for x in blocks:
    imgFl = x.astype(np.float32)
    value = np.sqrt(((imgFl.shape[0]/2.0)**2.0 + (imgFl.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(imgFl, (imgFl.shape[0]/2, imgFl.shape[1]/2), blockSize/2, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    blocksLog.append(polar_image)

#cv2.imshow("block log", blocksLog[5 + 5*blockPerLine])
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()


nbBlocks = len(blocksLog)
L = int(nbBlocks*(nbBlocks-1)/2)
listEquals = []
listMax = {}
listMean = {}

#adv = 1
'''
*************************************************
calcul g et f-1(g)
*************************************************
'''
for i in range(0, nbBlocks):
    G_a = np.fft.fft2(blocksLog[i])
    for j in range(i+1, nbBlocks):
        G_b = np.fft.fft2(blocksLog[j])

        conj_b = np.ma.conjugate(G_b)

        R = G_a*conj_b
        R /= np.absolute(R)
        r = np.fft.ifft2(R).real
        #nr = np.absolute(r)

        #_max = np.amax(nr)
        #_mean = np.mean(nr)

        listMax[i, j] = np.amax(r) #_max
        listMean[i, j] = np.mean(r) #_mean

        print("Mean/max du bloc [{}, {}] : {}/{}".format(i, j, listMean[i, j], listMax[i, j]))
        #print("progression : {}/{}".format(adv, L))
        #adv += 1

globMeanList = list(listMean.values())
globMean = np.mean(globMeanList)

#globMean = 0

'''
for i in range(0, nbBlocks):
    for j in range(i+1, nbBlocks):
        globMean += listMean[i, j]

_mean = globMean/L
'''


'''
for i in range(0, len(blocksLog)):
    for j in range(i+1, len(blocksLog)):
        if listMax[i, j] > 0.35: #globMean:
            listEquals.append([i, j])
'''

'''
*************************************************
test seuil
*************************************************
'''
for key, value in listMean.items():
    print("key : {}, value : {}".format(key, value))
    if value < 0.35:
        listEquals.append(key)


'''
*************************************************
affichage zones dupliquees
*************************************************
'''
i = 1
for x in listEquals:
    blocI = x[0]
    blocJ = x[1]

    xI = int((blocI % blockPerLine)) * blockSize
    yI = int((blocI/blockPerLine)) * blockSize

    xJ = int((blocJ % blockPerLine)) * blockSize
    yJ = int((blocJ/blockPerLine)) * blockSize

    img = cv2.rectangle(img, (xI, yI), (xI + blockSize - 1, yI + blockSize - 1), (0, 0, 255), -1)
    img = cv2.rectangle(img, (xJ, yJ), (xJ + blockSize - 1, yJ + blockSize - 1), (0, 0, 255), -1)
    #print("Highlight zone {}/{}".format(i, len(listEquals)))
    i += 1

cv2.imshow('imgFals', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

#print("Moyennes : {}".format(listMean))
#print("Moyenne : {}".format(globMean))


#print("Liste de couples : {}".format(listEquals))

