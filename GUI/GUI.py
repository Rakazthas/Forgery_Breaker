from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import showerror
import cv2.cv2 as cv2

'''
import les scripts de test
'''
global img

def openImg():
    res = askopenfilename(title="Ouvrir une image", filetypes=[('all files', '.*')])
    filePath.delete(0, END)
    filePath.insert(0, res)

def testForgery():
    img = cv2.imread(filename=path.get())
    if img is None:
        showerror("Erreur", "Veuillez entrer un chemin d'image valide")
    else:
        print("test en cours")

window = Tk()

imageWindow = PanedWindow(window, orient=HORIZONTAL)

path = StringVar()
path.set("Chemin de l'image")
filePath = Entry(imageWindow, textvariable=path)

imgButton = Button(imageWindow, text="Choisir une image", command=openImg)

imageWindow.add(filePath)
imageWindow.add(imgButton)

imageWindow.pack()

optionsWindow = PanedWindow(window, orient=HORIZONTAL)

eps = Spinbox(optionsWindow, from_=0, to=500)
minSamples = Spinbox(optionsWindow, from_=0, to=50)

optionsWindow.add(eps)
optionsWindow.add(minSamples)

optionsWindow.pack()


launchButton = Button(window, text="Lancer le test", command=testForgery)
launchButton.pack(side=BOTTOM)

window.mainloop()
