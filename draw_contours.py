import cv2 as cv
import numpy as np
import math
import os
import sys
import shutil
import xml.etree.ElementTree as ET

'''
Pre funkcnost programu musia byt subory ulozene podla nasledujucej struktury.
1. Vsetko je zabalene do priecinka imgs
2. Vsetky mensie obrazky a kontury patriace 1 obrazku su ulozene v osobitnom priecinku.
3. Tieto priecinky su ocislovane 0,1,2,...
4. V kazdom z tychto priecinkov su obrazky a .txt subory pomenovane 'a,b'
    a - suradnica y v ramci velkeho obrazka
    b - suradnica x v ramci velkeho obrazka
    Napr. 0,1.txt je XML subor s konturami k obrazku 0,1.img
Obrazky buniek budu ulozene do data/train/image.
Obrazky kontur budu ulozene do data/train/label.
Treningove obrazky budu ulozene do data/test 
'''

# PREMENNE
idx = 0
from_user = sys.argv
if (len(from_user) > 1): # od pouzivatela
    min_area = int(from_user[1])
    max_area = int(from_user[2])
    min_circ = float(from_user[3])
else: # default
    min_area = 0
    max_area = 200
    min_circ = 0.8

print('Minimum cell area: ' + str(min_area))
print('Maximum cell area: ' + str(max_area))
print('Minimum cell circularity: ' + str(min_circ))

# METODA NA KRESLENIE KONTUR
def drawContours(contours, filename):
    cont_img = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8) # white blank image
    for cnt in contours:
        cv.drawContours(cont_img, [cnt], 0, (0,0,0), 1)    
    cv.imwrite(filename, cont_img)

# u,v - poloha miniobrazka vramci obrazka
# od x,y treba odcitat okraje eX,eY a u,v
def makeContours(folder, filename, v, u):
    global idx
    global min_area
    global max_area
    global min_circ
    eX = eY = 30 # hranicne body

    contours = []
    contours_area = []
    contours_circ = []

    # PRECITAT XML & ULOZIT KONTURY
    root = ET.parse(folder+'/'+filename).getroot()
    for cont in root.findall('Contours/Contour'):
        contour = []
        for point in cont.findall('Point'):
            x = int(point.get('x')) - eX - u*512
            y = int(point.get('y')) - eY - v*512
            contour.append([x,y])
        contour = np.array(contour, dtype=np.int32) # zmenit na array
        contours.append(contour) # pridat medzi contours

        # filter podla plochy (area)
        area = cv.contourArea(contour)
        if area > min_area and area < max_area:
            contours_area.append(contour)

        # filter podla cirkularity
        perimeter = cv.arcLength(contour,True)
        circ = 0 if perimeter == 0 else 4*math.pi*(area/(perimeter*perimeter))
        if min_circ < circ < 1:
            contours_circ.append(contour)
            
    drawContours(contours, 'data/train/label/' + str(idx) + '.png')
    drawContours(contours_area, 'data/train/label_area/' + str(idx) + '.png')
    drawContours(contours_circ, 'data/train/label_circ/' + str(idx) + '.png')    


def readXMLfiles(folder):
    global idx
    for filename in os.listdir(folder):
        # subory cita v abecednom poradi, tj: 0,1.img, 0,1.txt, 0,2.img, 0,2.txt ...
        s1 = filename.split('.')
        if s1[1] == 'png':
            # subor obrazku - ulozime ho do image priecinku pod nazvom idx
            shutil.copyfile(folder + '/' + filename, 'data/train/image/' + str(idx) + '.png')
        else:
            # subor s konturami - vytvoreny obrazok s konturami ulozime do label priecinku pod nazvom idx
            s2 = s1[0].split(',')
            makeContours(folder, filename, int(s2[0]), int(s2[1]))
            idx += 1


# PRIPRAVIT STRUKTURU PRIECINKOV
def createFolders():
    os.makedirs('data/train/image')
    os.mkdir('data/train/label')
    os.mkdir('data/train/label_area')
    os.mkdir('data/train/label_circ')
    os.mkdir('data/test')

def moveTestImages():
    for filename in os.listdir('imgs/test'):
        shutil.copyfile('imgs/test/' + filename, 'data/test/' + filename)

def convertTo8b(folder):
    for filename in os.listdir(folder):
        img = cv.imread(folder+'/'+filename)
        gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imwrite(folder+'/'+filename,gray)

# VYTVORIT OBRAZKY BUNIEK A KONTUR
createFolders()
for folder in range (6):
    readXMLfiles('imgs/' + str(folder)) # priecinky su nazvane 0,...5
    print('Image ' + str(folder) + ' of 5 proceeded')
moveTestImages()

# previest obrazky na ciernobiele 8b
convertTo8b('data/train/image')
convertTo8b('data/train/label')
convertTo8b('data/train/label_area')
convertTo8b('data/train/label_circ')
convertTo8b('data/test')