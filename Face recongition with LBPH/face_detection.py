import cv2
import numpy as np
import pickle


faceCascade = cv2.CascadeClassifier("C:\\Users\\Htet\\Desktop\\Face Recognition LBPH\\all new\\data\\haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}

with open("C:\\users\\Htet\\Desktop\\Face Recognition LBPH\\all new\\labels.pickle",'rb') as f:
	og_labels = pickle.load(f)
	print(og_labels)

	labels = {v:k for k,v in og_labels.items()}
	print(labels)



cap = cv2.VideoCapture(0)


while True:
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

	for (x,y,w,h) in faces:

		roi_gray = gray[y:y+h, x:x+w]

		id_, conf = recognizer.predict(roi_gray)
		if conf >= 45:# and conf <= 85:
			print(id_)
			print(labels[id_])

			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


		cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,255),2)



	cv2.imshow("Frame", frame)
	if cv2.waitKey(1)& 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()