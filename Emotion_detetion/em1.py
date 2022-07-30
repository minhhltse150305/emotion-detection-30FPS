import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp
import time

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)
opt = tf.keras.optimizers.Adam(0.0001, decay=1e-6)

emotion_model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])
emotion_model = keras.models.load_model('model9.h5')
cap = cv2.VideoCapture(0)
pTime=0

mpFaceDetecsion = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetecsion.FaceDetection(0.25)

def fancyDraw(img , bbox ,l = 30 , t = 5 , rt =1):
    x , y ,w ,h = bbox
    x1 , y1 = x+w, y+h
    #top left
    cv2.line(img , (x,y),(x+l , y), (255,0,255) ,t)
    cv2.line(img, (x, y), (x , y+l), (255, 0, 255), t)
    # top right
    cv2.line(img, (x1, y), (x1 -l , y ), (255, 0, 255), t)
    cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
    #bot left
    cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
    cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
    # bot left
    cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
    return img


def emotion(img,result):
    if result.detections:
        for id, detection in enumerate(result.detections):
            #print(id, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih , iw , ic = img.shape
            bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox , (255,0,255),1)
            img = fancyDraw(img , bbox)
            #cv2.putText(img , f'{int(detection.score[0]*100)}%',
            #            (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
            #            3,(255,0,255),2)
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, iw)
            y2 = min(y + h, ih)

            face = img[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            final_image1 = cv2.resize(face, (48, 48))
            final_image1 = np.expand_dims(final_image1, axis=0)
            final_image1 = final_image1 / 255.0
            Pridictions = emotion_model.predict(final_image1)
            print(np.argmax(Pridictions))
            if (np.argmax(Pridictions) == 0):
                status = "Angry"
                cv2.putText(img, status ,
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 0, 255), 2)

            elif (np.argmax(Pridictions) == 1):
                status = "Disgust"
                cv2.putText(img, status, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 255), 2)

            elif (np.argmax(Pridictions) == 2):

                status = "Fear"
                cv2.putText(img, status,
                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255), 2)
            elif (np.argmax(Pridictions) == 3):

                status = "Happy"
                cv2.putText(img, status, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
            elif (np.argmax(Pridictions) == 4):
                status = "Neutrual"
                cv2.putText(img, status, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
            elif (np.argmax(Pridictions) == 5):

                status = "Sad"
                cv2.putText(img, status, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)


            else:
                status = "Suprise"
                cv2.putText(img, status, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

while True:
    success , img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgRGB)

    emotion(img ,result)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),2)
    cv2.imshow("image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()