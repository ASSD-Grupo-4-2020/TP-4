'''

Este Archivo reconoce rostros en relacion a la base de datos utilizada

'''

import cv2
import pickle
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')


labels = {'person_name': 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k,v in og_labels.items()}


#Aca se coloca el path hacia la imagen que desee identificar
original_image = cv2.imread('Test_images/test1.jpeg')
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

detected_faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.5, minNeighbors=5)

for x, y, w, h in detected_faces:
    roi_gray = grayscale_image[y:y + h, x:x + w]

    id_, conf = recognizer.predict(roi_gray)
    if conf <= 70:

        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_] + '  ' + str(conf)
        color = (255, 155, 0)
        stroke = 3
        cv2.putText(original_image, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

    elif conf >= 70:
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = 'No Idea' + '  ' + str(math.trunc(conf))
        color = (255, 155, 0)
        stroke = 1
        cv2.putText(original_image, name, (x, y), font, 0.5, color, stroke, cv2.LINE_AA)

    img_item = 'my_image.png'
    cv2.imwrite(img_item, roi_gray)

    rectangle_color = (255, 155, 0)
    rectangle_thickness = 3
    width = x + w
    height = y + h

    cv2.rectangle(original_image, (x, y), (width, height), rectangle_color, rectangle_thickness)

cv2.imshow('frame', original_image)

cv2.waitKey(0)
cv2.destroyAllWindows()