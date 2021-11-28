""" 
Face detector using OpenCV and Python and pre-trained models with HaarCascade and IBP Cascades of frontalfaces 
Copyright (C) 2021 Gerald Vargas Vásquez, Nelson Vega Soto

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses.

Face detector using OpenCV and Python Copyright (C) 2021 Gerald Vargas Vásquez, Nelson Vega Soto
This program comes with ABSOLUTELY NO WARRANTY; 
This is free software, and you are welcome to redistribute it
under certain conditions.

"""

import cv2 as cv
import os
import imutils

# Obtenido del repositorio código abierto de OpenCV https://github.com/opencv/opencv/tree/4.x/data
haarcasde = cv.CascadeClassifier('haarcascade.xml')
ibpcascade = cv.CascadeClassifier('ibpcascade.xml')
   
def video_processing(classifier,path):
    capture = cv.VideoCapture(0,cv.CAP_DSHOW)
    faceCount = 0
    while True :
        ret,frame = capture.read()
        if ret == False: break
        frame =  imutils.resize(frame, width=640)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        extractor = classifier.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in extractor:
            cv.rectangle(0,(x,y),(x+w , y+h),(0,255,0),2)
            face = auxFrame[y:y+h,x:x+w]
            face = cv.resize(face,(150,150),interpolation=cv.INTER_CUBIC)
            cv.imwrite(path + '/face_{}.jpg'.format(faceCount),face)
            faceCount = faceCount + 1
        cv.imshow('frame',frame)
        
        k =  cv.waitKey(1)
        if k == 27 or faceCount >= 200:
            break
    

def main(classfier,person,emotion):
    path = './Data/'+person+'/'+emotion
    if not os.path.exists(path):
        os.makedirs(path)
    video_processing(classfier,path)

main(haarcasde,'Gerald','Feliz')


