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

#text=input("Entrez un text")
#img =Image.open(text)
img =Image.open("14_Perroquet.ppm")
w,h =img.size




#log polar
#L'image doit être en RGB si multichannel =True
img=cv2.imread("Perroquet2.ppm")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
radius = w/2
angle = 35
imglp = img_as_float(img)
rotated = rotate(imglp, angle)
image_polar = warp_polar(imglp, radius=radius, multichannel=False)
rotated_polar = warp_polar(rotated, radius=radius, multichannel=False)

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


#imgage_polar=cv2.cvtColor(image_polar,cv2.COLOR_RGB2GRAY)
f=np.fft.fft2(image_polar)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(image_polar, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

f=np.fft.fft2(rotated_polar)
fshift = np.fft.fftshift(f)
magnitude_spectrum_polar = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Input Image MS'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum_polar, cmap = 'gray')
plt.title('Rotated Image MS'), plt.xticks([]), plt.yticks([])
plt.show()


