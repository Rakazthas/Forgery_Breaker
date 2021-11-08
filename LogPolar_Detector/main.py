# Log-Polar detection script
# Base imports
import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
from math import copysign,log10

# Log-Polar operations
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float


imgName = input("Entrer le chemin de l'image : ")
img = cv2.imread(imgName)
h,w = img.shape[:2]

print("Format des pixels: {}".format(img.mode))
print("Largeur : {} px,hauteur:{} px".format(w,h))
px_value =img.getpixel((20,100))
print("Valeur du pixel en (20,100): {}".format(px_value))
cv2.imshow('img', img)