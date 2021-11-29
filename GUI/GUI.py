from tkinter import *
from tkinter.filedialog import *
'''
import les scripts de test
'''


def openImg():
    res = askopenfilename(title="Ouvrir une image", filetypes=[('all files', '.*')])
    filePath.delete(0, END)
    filePath.insert(0, res)


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

optionsWindow.pack()


launchButton = Button(window, text="Lancer le test")
launchButton.pack(side=BOTTOM)

window.mainloop()
