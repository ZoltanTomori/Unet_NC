import cv2 as cv
import os
import sys


def thresholdLabeledImages(folder):
    idx = 0
    for filename in os.listdir(folder):
        if len(filename.split('_')) > 1: # len obrazky _predict chceme previest
            img = cv.imread(folder + '/' + filename,0)

            # obrazok ma rozmer 256 x 256
            for i in range(256):
                for j in range(256):
                    if img[i,j] > 256*0.9: # prah = 256*0.9
                        img[i,j] = 255
                    else:
                        img[i,j] = 0       
            cv.imwrite(folder + '/' + str(idx) + 't.png', img)
            idx += 1


if (len(sys.argv) > 1):
    thresholdLabeledImages(sys.argv[1]) # nazov priecinka od pouzivatela
else:
    thresholdLabeledImages('predicted')
print('Thresholding finished.')