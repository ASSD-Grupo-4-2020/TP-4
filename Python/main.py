import numpy as np
import cv2
import pickle



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')



labels = {'person_name': 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        #until here we have found a face in the frame.
        #recognize -----> who is that person??

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 25:


            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_] + '  ' + str(conf)
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
    cv2.imshow('frame', frame)


    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()