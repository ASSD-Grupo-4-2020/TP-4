import numpy as np
import cv2
import pickle
from Python.Fps import FPS, WebcamVideoStream


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')



labels = {'person_name': 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k,v in og_labels.items()}


cap = WebcamVideoStream(src=0).start()
fps = FPS().start()

while True:
    frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        #until here we have found a face in the frame.
        #recognize -----> who is that person??

        id_, conf = recognizer.predict(roi_gray)
        if conf <= 70:


            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_] + '  ' + str(conf)
            color = (255, 155, 0)
            stroke = 3
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        elif conf >= 70:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = 'No Idea' + '  ' + str(conf)
            color = (255, 155, 0)
            stroke = 3
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)




        img_item = 'my_image.png'
        cv2.imwrite(img_item, roi_gray)

        rectangle_color = (255, 155, 0)
        rectangle_thickness = 3
        width = x + w
        height = y + h

        cv2.rectangle(frame, (x, y), (width, height), rectangle_color, rectangle_thickness)

        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for ex, ey, ew, eh in eyes:
        #    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        #smile = smile_cascade.detectMultiScale(roi_gray)
        #for sx, sy, sw, sh in smile:
        #    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)



    cv2.imshow('frame', frame)
    fps.update()

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

fps.stop()
cv2.destroyAllWindows()
cap.stop()