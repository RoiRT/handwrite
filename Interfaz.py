from PyQt5.QtCore import QFile
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QLabel, QScrollArea, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from unidecode import unidecode
import numpy as np
from keras.models import load_model
from PIL import Image, ImageFilter
from variables import caracteres

import sys

def leeInterfaz(fichero):
    archivoUi = QFile(fichero)
    archivoUi.open(QFile.ReadOnly)
    ventana = uic.loadUi(archivoUi)
    archivoUi.close()
    return ventana

aplicacion = QApplication(sys.argv)
ventana = leeInterfaz("Interfaz.ui")

generator = load_model('generator.h5')
textEdit = ventana.findChild(QTextEdit, "textEdit")
button = ventana.findChild(QPushButton, "button")
label = ventana.findChild(QLabel, "label")

button_save = ventana.findChild(QPushButton, "button_save")
scrollArea = ventana.findChild(QScrollArea, "scrollArea")
image = ventana.findChild(QLabel, "image")
next_button = ventana.findChild(QPushButton, "next_button")
previous_button = ventana.findChild(QPushButton, "previous_button")
page_num = ventana.findChild(QLabel, "page_num")


image.setScaledContents(True)
textEdit.setLineWrapMode(QTextEdit.WidgetWidth)
textEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

button_save.setVisible(False)
image.setVisible(False)
scrollArea.setVisible(False)
next_button.setVisible(False)
previous_button.setVisible(False)
page_num.setVisible(False)

scrollArea.setWidget(image)
scrollArea.setWidgetResizable(True)
scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)


def generateLetters(text):
    letters = np.array(list(text.replace(" ", ""))).reshape(-1, 1)

    caracteres_invert = {j: i for i, j in caracteres.items()}
    caracteres_invert.update(
        {'c': 12, 'i': 18, 'j': 19, 'k': 20, 'l': 18, 'm': 22, 'o': 24, 'p': 25, 's': 28, 'u': 30, 'v': 31, 'w': 32,
         'x': 33, 'y': 34, 'z': 35})

    letters_input = np.vectorize(caracteres_invert.get)(letters)

    letters_pics = []
    for i in range(10):
        noise = np.random.normal(0, 1, [len(letters_input), 100])
        letters_pics.append(generator.predict([noise, letters_input]))

    return letters_pics

def generateImage(letters_pics, space_position):
    ancho_a4 = 2100
    alto_a4 = 2970

    papers_a4 = []
    for i in range(10):
        papers_a4.append(Image.new("L", (ancho_a4, alto_a4), "white"))

    origen_x = 60
    origen_y = 100
    pos = 0
    p = [origen_x, origen_y]
    for i in range(space_position[-1]):
        if i in space_position:
            if (ancho_a4 - origen_x - p[0]) < (space_position[1] - i)*28:
                p[1] += 60
                p[0] = origen_x
            else:
                p[0] += 28
            space_position.pop(0)
        else:
            position = (p[0], p[1])
            for j in range(10):
                letter = letters_pics[j][pos]
                photo = Image.fromarray((255 - letter.squeeze()*255).astype(np.uint8), mode="L").filter(ImageFilter.EDGE_ENHANCE)
                papers_a4[j].paste(photo.filter(ImageFilter.MedianFilter(size=3)), position)
            p[0] += 28
            pos += 1
    return papers_a4

def clickedButtonNext():
    global current_pixmap
    if current_pixmap < len(pixmaps) - 1:
        current_pixmap += 1
    else:
        current_pixmap = 0

    pasteImage()

def clickedButtonPrevious():
    global current_pixmap

    if current_pixmap > 0:
        current_pixmap += 1
    else:
        current_pixmap = len(pixmaps) - 1

    pasteImage()

def pasteImage():
    image.setVisible(True)
    scrollArea.setVisible(True)
    button_save.setVisible(True)
    next_button.setVisible(True)
    previous_button.setVisible(True)
    page_num.setVisible(True)
    image.setPixmap(pixmaps[current_pixmap])
    page_num.setText(str(current_pixmap+1) + "/10")


def clickedButtonSave():
    global current_pixmap

    options = QFileDialog.Options()
    filename, _ = QFileDialog.getSaveFileName(None, "Guardar Imagen", "", "Archivos de Imagen (*.png *.jpg)", options=options)

    if filename:
        papers[current_pixmap].save(filename)

def clickedButton():
    global current_pixmap, papers, pixmaps
    current_pixmap = 0

    text = textEdit.toPlainText().rstrip()

    if text.strip() == "":
        label.setText("El campo está vacío")
    elif not text.replace(" ", "").isalnum():
        label.setText("Solo se permiten letras y números")
    elif len(max(text.split(), key=len)) > 20:
        label.setText("no puede haber palabras de más de 20 caracteres")
    elif len(text.split()) > 250 or len(text) > 2000:
        label.setText("Solo se pueden añadir como máximo 250 palabras o 2000 caracteres")
    else:
        label.setText("Texto válido, espere...")
        QApplication.processEvents()
        letters_pics = generateLetters(unidecode(text))

        space_position = [i for i, x in enumerate(list(text)) if x == ' ']
        papers = generateImage(letters_pics, space_position + [len(text)])

        pixmaps = []
        for i in range(10):
            image_qt = QImage(papers[i].tobytes(), papers[i].size[0], papers[i].size[1], QImage.Format_Grayscale8)
            pixmaps.append(QPixmap.fromImage(image_qt))

        pasteImage()

        label.setText("Listo!")


button.clicked.connect(clickedButton)
button_save.clicked.connect(clickedButtonSave)
previous_button.clicked.connect(clickedButtonPrevious)
next_button.clicked.connect(clickedButtonNext)

ventana.show()
sys.exit(aplicacion.exec_())

