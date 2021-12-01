# Forgery_Breaker
Scripts to detect forgery in an image, using SIFT and clustering approach for copy-move forgery, and neural network for splicing.

---
#Libraries needed

    -Opencv
    -Numpy
    -Sklearn
    -Keras
    -Matplotlib
---
#Usage
##Copy-move
Run GUI.py in GUI folder, choose an image, and adjust eps and minSamples to your image, then press "run test".

    -eps : max distance for neighbour research (from 1 to 500)
    -minsamples : minimum number of neighbours required for a cluster (from 2 to 50)