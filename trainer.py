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
import numpy as np

# Obtenido del repositorio código abierto de OpenCV https://github.com/opencv/opencv/tree/4.x/data
haarcasde = cv.CascadeClassifier('./Models/haarcascade.xml')
ibpcascade = cv.CascadeClassifier('./Models/ibpcascade.xml')
   
def videoProcessing(classifier,person,emotion):
    path = './Data/'+person+'-'+emotion
    if not os.path.exists(path):
        os.makedirs(path)

    capture = cv.VideoCapture(0,cv.CAP_DSHOW)
    faceCount = 0
    while True :
        ret,frame = capture.read()
        if ret == False: break
        frame =  imutils.resize(frame, width=640)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        extractor = classifier.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in extractor:
            cv.rectangle(0,(x,y),(x+w , y+h),(0,255,0),2)
            face = frame[y:y+h,x:x+w]
            face = cv.resize(face,(150,150),interpolation=cv.INTER_CUBIC)
            cv.imwrite('./Data/'+person+'-'+emotion +'/face_{}.jpg'.format(faceCount),face)
            faceCount = faceCount + 1
        cv.imshow('frame',frame)
        
        k =  cv.waitKey(1)
        if k == 27 or faceCount >= 200:
            break           

def facesProcessing(person):
    labels = []
    faces = []
    label = 0
    for person in os.listdir('./Data/'):
        for emotion in os.listdir('./Data/'+person):
            for fileName in os.listdir('./Data/'+person+"-"+emotion):
                labels.append(label)
                faces.append(cv.imread('./Data/'+person+"-"+emotion+'/'+fileName,0))
            label = label + 1
    trainModel('EigenFaces',faces,labels)
    trainModel('LBPH',faces,labels)

# Method    :   1 = EigenFaces  2 = LBPH
# Path      :   ./Data/Person/Emotion                                                         
def trainModel(method,faces,labels):
    if method == 'EigenFaces' :
        recognizer = cv.face.EigenFaceRecognizer_create()
    if method == 'LBPH' :
        recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(labels))
    recognizer.write("./Models/"+method+".xml")


videoProcessing(haarcasde,'Andrey','Enojado')
#videoProcessing(haarcasde,'Gerald','Feliz')
#videoProcessing(haarcasde,'Gerald','Triste')
#videoProcessing(haarcasde,'Gerald','Neutro')

#facesProcessing('Gerald')


