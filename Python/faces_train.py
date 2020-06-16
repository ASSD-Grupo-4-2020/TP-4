import cv2
import os
from PIL import Image
import numpy as np
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}

y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
            path = os.path.join(root, file)

            label = os.path.basename(root).replace(' ', '-').lower()


            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id = label_ids[label]


            pil_image = Image.open(path).convert('L') #turns into grayscale

            size = (550, 500)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            image_array = np.array(final_image, 'uint8') #image turned into numbers

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for x, y, w, h in faces:
                roi = image_array[y: y + h, x: x + w]
                x_train.append(roi)
                y_labels.append(id)


with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)



#Training opencv recognizer (lbph face recognizer)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')
