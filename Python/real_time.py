'''

Este archivo sirve para reconocimiento de video en tiempo real

'''


import cv2
import pickle
from Python.Fps import FPS, WebcamVideoStream
import math

#creo clasificadores
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


#leo el reconocedor que habia creado y entrenado anteriormente
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml') #este es el reconocedor entrenado


labels = {'person_name': 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

#captura de video opencv
cap = WebcamVideoStream(src=0).start()
fps = FPS().start()


while True:
    frame = cap.read()

    #busco rostros en un frmae dado
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #faces contiene los rostros encontrados en el frmae de video mediante el algoritmo viola jones

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        #hasta aca solo encontramos un rostro y no lo reconocemos

        id_, conf = recognizer.predict(roi_gray)

        #defino la confianza para que reconozca o no a la persona
        if conf <= 70:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_] + '  ' + str(math.trunc(conf))
            color = (255, 155, 0)
            stroke = 3
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        elif conf >= 70:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = 'No Idea' + '  ' + str(math.trunc(conf))
            color = (255, 155, 0)
            stroke = 3
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        #El codigo anterior pone el nombre de la persona en los marcos si reconoce a alguna persona de la abse de datos

        img_item = 'my_image.png'
        cv2.imwrite(img_item, roi_gray)

        rectangle_color = (255, 155, 0)
        rectangle_thickness = 3
        width = x + w
        height = y + h

        cv2.rectangle(frame, (x, y), (width, height), rectangle_color, rectangle_thickness)

    #muestro el marco
    cv2.imshow('frame', frame)
    fps.update()

    #Presione q para cerrar el programa
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


fps.stop()
cv2.destroyAllWindows()
cap.stop()
