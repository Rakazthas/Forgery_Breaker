import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from PIL import Image
from math import copysign,log10
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

img =Image.open("14_Perroquet.ppm")
w,h =img.size
print("Format des pixels: {}".format(img.mode))
print("Largeur : {} px,hauteur:{} px".format(w,h))
px_value =img.getpixel((20,100))
print("Valeur du pixel en (20,100): {}".format(px_value))
img.show()

mat = np.array(img)

#print(mat)
"""
print("Taille de la matrice de pixels : {}".format(mat.shape))
"""

#Histo

n, bins, patches = plt.hist(mat.flatten(), bins=range(256))
plt.show()



#si RGB -> ndg

img=img.convert('L')
img.show()


#DCT from scipy

img2=dct(dct(img, norm='ortho').T,norm='ortho')
img2=Image.fromarray((img2))
img2.show()

#HuMoments from opencv

showLogTransformedHuMoments = True
im=cv2.imread("14_Perroquet.ppm",cv2.IMREAD_GRAYSCALE)
moment=cv2.moments(im)
huMoments=cv2.HuMoments(moment)

for i in range(0,7):
            if showLogTransformedHuMoments:
                # Log transform Hu Moments to make
                # squash the range
                print("{:.5f}".format(-1*copysign(1.0,\
                        huMoments[i])*log10(abs(huMoments[i]))),\
                        end=' ')
            else:
                # Hu Moments without log transform
                print("{:.5f}".format(huMoments[i]),end=' ')

#log polar
#L'image doit être en RGB si multichannel =True
img =Image.open("14_Perroquet.ppm")
radius = 705
angle = 35
imglp = img_as_float(img)
rotated = rotate(imglp, angle)
image_polar = warp_polar(imglp, radius=radius, multichannel=True)
rotated_polar = warp_polar(rotated, radius=radius, multichannel=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original")
ax[0].imshow(imglp)
ax[1].set_title("Rotated")
ax[1].imshow(rotated)
ax[2].set_title("Polar-Transformed Original")
ax[2].imshow(image_polar)
ax[3].set_title("Polar-Transformed Rotated")
ax[3].imshow(rotated_polar)
plt.show()

#SVD


img = np.mean(img, 2)

U,s,V = np.linalg.svd(img)

n = 10
S = np.zeros(np.shape(img))
for i in range(0, n):
    S[i,i] = s[i]

recon_img = U @ S @ V

fig, ax = plt.subplots(1, 2)

ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(recon_img)
ax[1].axis('off')
ax[1].set_title(f'Reconstructed n = {n}')

plt.show()

#Harris corner Detection
#Doit être en ndg
filename = '14_Perroquet.ppm'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()