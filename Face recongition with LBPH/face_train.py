import os
import numpy as np
import cv2
from PIL import Image
import pickle



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, "Face Images")


faceCascade = cv2.CascadeClassifier("C:\\Users\\Htet\\Desktop\\all new\\data\\haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)

			label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
			#print(path)
			#print(label)

			#y_labels.append(label)#some number for labels
			#x_train.append(path)#varify this image, turn it into a numpy array


			if not label in label_ids:
				label_ids[label] = current_id
				current_id +=1

			id_ = label_ids[label]
			#print(id_)
			#print(label_ids)





			#changing into numpy array

			pil_image = Image.open(path).convert('L')#gray scale
			image_array = np.array(pil_image,"uint8")

			#print(image_array)

			faces = faceCascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for(x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
#print(y_labels)

#print(np.array(y_labels))


with open("labels.pickle",'wb') as f:
	pickle.dump(label_ids,f)



recognizer.train(x_train, np.array(y_labels))

recognizer.save("trainner.yml")