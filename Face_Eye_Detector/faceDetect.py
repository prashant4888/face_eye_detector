import cv2
import numpy as np 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)  #Set the index to 1 if not using external webcam
while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)

	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0), 2)
		cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		eyes = eye_cascade.detectMultiScale(roi_gray)

		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
			cv2.putText(roi_color, "Eye", (ex, ey), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cap.release()
cv2.destroyAllWindws()
