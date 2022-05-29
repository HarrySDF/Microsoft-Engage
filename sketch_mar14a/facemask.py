import cv2, datetime
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.SerialModule import SerialObject

import numpy as np
import keras
from keras.preprocessing import image

# Loading the model
model = keras.models.load_model(r'C:\Users\Admin\PycharmProjects\FaceMask\face_detection_mask\face_detection_mask')

# opening camera for live detection
cap = cv2.VideoCapture(0)

# loading haarcascade file for detecting face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# creating object of arduino
arduino = SerialObject()


while cap.isOpened():
    _, img = cap.read()

    # face detected
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in face:
        face_img = img[y:y + h, x:x + w]

        # saving the image
        cv2.imwrite('temp.jpg', face_img)

        # loading the image and pre-processed
        test_image = keras.utils.load_img('temp.jpg', target_size=(64, 64, 3))
        test_image = keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # predicting the result
        pred = model.predict(test_image)[0][0]

        # No mask condition
        if pred == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # sending data to arduino
            arduino.sendData([1]);

        # yes mask condition
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # sending data to arduino
            arduino.sendData([0]);

        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('img', img)

    # exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
