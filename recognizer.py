import cv2 as cv
import os
import sys

def recognizer(method):
	
	if method == 'EigenFaces': 
		if not os.path.exists('./Models/EigenFaces.xml'):
			print('No existe un modelo para ejecutar el reconocedor')
			sys.exit()
		else:
			recognizer = cv.face.EigenFaceRecognizer_create()
			recognizer.read('./Models/EigenFaces.xml')
	if method == 'LBPH': 
		if not os.path.exists('./Models/LBPH.xml'):
			print('No existe un modelo para ejecutar el reconocedor')
			sys.exit()
		else: 
			recognizer = cv.face.LBPHFaceRecognizer_create()
			recognizer.read('./Models/EigenFaces.xml')
		
	path = './Data/'
	imagePaths = os.listdir(path)
	cap = cv.VideoCapture(0,cv.CAP_DSHOW)
	faceClassif  = cv.CascadeClassifier('./Models/haarcascade.xml')
	
	while True:

		ret,frame = cap.read()
		if ret == False: break
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		auxFrame = gray.copy()
		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			face = auxFrame[y:y+h,x:x+w]
			face = cv.resize(face,(150,150),interpolation= cv.INTER_CUBIC)
			result = recognizer.predict(face)

			if method == 'EigenFaces':
				cv.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
				cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			
			if method == 'LBPH':
				cv.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
				cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				
		cv.imshow('Streaming',frame)
		k = cv.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv.destroyAllWindows()


recognizer('EigenFaces')
	



