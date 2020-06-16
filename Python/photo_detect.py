import cv2


def face_detection(decision):
    original_image = cv2.imread('images/Multiples/multiple_1.jpg')
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    detected_faces2 = face_cascade2.detectMultiScale(grayscale_image)

    if decision:
        for (column, row, width, height) in detected_faces:
            cv2.rectangle(
                original_image,
                (column, row),
                (column + width, row + height),
                (0, 255, 0),
                2
            )

        cv2.imshow('Image', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        for (column, row, width, height) in detected_faces2:
            cv2.rectangle(
                original_image,
                (column, row),
                (column + width, row + height),
                (0, 255, 0),
                2
            )

        cv2.imshow('Image', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


face_detection(0)
