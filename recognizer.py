import cv2
import os
import numpy as np

def recognizer(method,person,source,url):
	
	if method == 'EigenFaces': 
		if not os.path.exists('./Models/EigenFaces.xml'):
			print('No existe un modelo para ejecutar el reconocedor')
		else:
			recognizer = cv2.face.EigenFaceRecognizer_create()
			recognizer.read('./Models/EigenFaces.xml')
	if method == 'LBPH': 
		if not os.path.exists('./Models/LBPH.xml'):
			print('No existe un modelo para ejecutar el reconocedor')
		else: 
			recognizer = cv2.face.LBPHFaceRecognizer_create()
			recognizer.read('./Models/EigenFaces.xml')
		
	path = './Data/'+person
	imagePaths = os.listdir(path)
	cap = ''
	if source == 'stream':
		cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

	classifier = cv2.CascadeClassifier('./Models/haarcascade.xml')
	
	while True:

		ret,frame = cap.read()
		if ret == False: break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()
		faces = recognizer.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
			result = recognizer.predict(rostro)

			cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

			if method == 'EigenFaces':
				if result[1] < 5700:
					cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				else:
					cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
			
			if method == 'LBPH':
				if result[1] < 60:
					cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				else:
					cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

		cv2.imshow('nFrame',frame)
		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()


	
# --------------------------------------------------------------------------------



