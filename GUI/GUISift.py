from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import showerror, showinfo
import cv2.cv2 as cv2

from siftDetector import SIFT


global img


def openImg():
    res = askopenfilename(title="Open file", filetypes=[('all files', '.*')])
    filePath.delete(0, END)
    filePath.insert(0, res)


def testForgery():
    img = cv2.imread(filename=path.get())
    if img is None:
        showerror("Error", "Please use a valid image")
    else:
        cv2.imshow("Original image", img)
        print("Test en cours")

        if epsVar.get() > 500:
            epsVar.set(500)
        if epsVar.get() <= 0:
            epsVar.set(1)

        if minSamplesVar.get() > 50:
            minSamplesVar.set(50)
        if minSamplesVar.get() < 2:
            minSamplesVar.set(2)

        detect = SIFT(img=img)
        detect.siftFind()

        features = detect.siftShowFeatures()
        cv2.imshow("Features", features)

        forgery = detect.forgeryLocate(eps=float(epsVar.get()), minSamples=float(minSamplesVar.get()))
        if forgery is None:
            showinfo("Result", "No copy-move forgery found")
        else:
            cv2.imshow("Forgery found", forgery)


window = Tk()

imageWindow = PanedWindow(window, orient=HORIZONTAL)

path = StringVar()
path.set("Image path")
filePath = Entry(imageWindow, textvariable=path)

imgButton = Button(imageWindow, text="Choose image", command=openImg)

imageWindow.add(filePath)
imageWindow.add(imgButton)

imageWindow.pack()

optionsWindow = PanedWindow(window, orient=HORIZONTAL)

epsVar = DoubleVar(value=40)
minSamplesVar = DoubleVar(value=2)

eps = Spinbox(optionsWindow, from_=1, to=500, textvariable=epsVar)
minSamples = Spinbox(optionsWindow, from_=2, to=50, textvariable=minSamplesVar)

optionsWindow.add(eps)
optionsWindow.add(minSamples)

optionsWindow.pack()

launchButton = Button(window, text="Run test", command=testForgery)
launchButton.pack(side=BOTTOM)

window.mainloop()
