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
emotion_model = keras.models.load_model('model9 (1).h5')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
frame = cv2.imread('images.jpg')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
  roi_gray = gray[y:y+h , x:x+h]
  roi_color = frame[y:y+h , x:x+h]
  cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
  facess = faceCascade.detectMultiScale(roi_gray)
  if len(facess) == 0:
    print('face not dection')
  else:
    for (ex,ey,ew,eh) in facess:
      face_roi1 = roi_color[ey: ey+eh, ex:ex+ew]


face_ro = cv2.cvtColor(face_roi1, cv2.COLOR_BGR2GRAY)
final_image1 = cv2.resize(face_ro, (48,48))
final_image1 = np.expand_dims(final_image1, axis =0)
final_image1 = final_image1 / 255.0
Pridictions = emotion_model.predict(final_image1)

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
#set the racteangle backgou to white
rectangle_bgr = (255,255,255)
#mak a back img
img = np.zeros((500,500))
#set some text
text = "some text in a box"
#get the width and height of the text box!
(text_width, text_height) = cv2.getTextSize(text, font, fontScale = font_scale, thickness=1)[0]
#set the text star position
text_offset_x = 10
text_offset_y = img.shape[0]-25

#make the coords of the box with a small padding of 2 pixels
box_coords = ((text_offset_x,text_offset_y), (text_offset_x+ text_width+2, text_offset_y- text_height -2))
cv2.rectangle(img , box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img , text , (text_offset_x,text_offset_y), font , fontScale = font_scale, color = (0,0,0), thickness=1)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
while True:
  ret , frame = cap.read()
  faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,1.1,4)
  for x,y,w,h in faces:
    roi_gray = gray[y:y+h , x:x+h]
    roi_color = frame[y:y+h , x:x+h]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
      print('face not dection')
    else:
      for (ex,ey,ew,eh) in facess:
        face_roi = roi_color[ey: ey+eh, ex:ex+ew]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        final_image = cv2.resize(face_roi, (48, 48))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0
        Priedictions = emotion_model.predict(final_image)
        font = cv2.FONT_HERSHEY_SIMPLEX

        font = cv2.FONT_HERSHEY_PLAIN

        font_scale = 1.5
        if (np.argmax(Priedictions) == 0):
          status = "Angry"
          x1, y1, w1, h1 = 0, 0, 175, 75
          # draw black backgroupd rectangle
          cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
          # add text
          cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(51 / 2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
                      (0, 0, 255), 2)
          cv2.putText(frame, status, (x,y), font, 3, (0, 0, 255), 2, cv2.LINE_4)

          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Priedictions) == 1):
          status = "Disgust"
          x1, y1, w1, h1 = 0, 0, 175, 75
          # draw black backgroupd rectangle
          cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
          # add text
          cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(51 / 2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
                      (0, 0, 255), 2)
          cv2.putText(frame, status, (x, y), font, 3, (0, 0, 255), 2, cv2.LINE_4)

          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Priedictions) == 2):
          status = "Fear"
          x1, y1, w1, h1 = 0, 0, 175, 75
          # draw black backgroupd rectangle
          cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
          # add text
          cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(51 / 2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
                      (0, 0, 255), 2)
          cv2.putText(frame, status, (x, y), font, 3, (0, 0, 255), 2, cv2.LINE_4)

          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Priedictions) == 3):
          status = "Happy"
          x1, y1, w1, h1 = 0, 0, 175, 75
          # draw black backgroupd rectangle
          cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
          # add text
          cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(51 / 2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
                      (0, 0, 255), 2)
          cv2.putText(frame, status, (x, y), font, 3, (0, 0, 255), 2, cv2.LINE_4)

          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Priedictions) == 4):
          status = "Neutrual"

          x1, y1, w1, h1 = 0, 0, 175, 75
          # draw black backgroupd rectangle
          cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
          # add text
          cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(51 / 2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
                      (0, 0, 255), 2)
          cv2.putText(frame, status, (x, y), font, 3, (0, 0, 255), 2, cv2.LINE_4)

          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        elif (np.argmax(Priedictions) == 5):
          status = "Sad"


          x1, y1, w1, h1 = 0, 0, 175, 75
          # draw black backgroupd rectangle
          cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
          # add text
          cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(51 / 2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
                      (0, 0, 255), 2)
          cv2.putText(frame, status, (x, y), font, 3, (0, 0, 255), 2, cv2.LINE_4)

          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        else:
          status = "Suprise"

          x1, y1, w1, h1 = 0, 0, 175, 75
          # draw black backgroupd rectangle
          cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
          # add text
          cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(51 / 2)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
                      (0, 0, 255), 2)
          cv2.putText(frame, status, (x, y), font, 3, (0, 0, 255), 2, cv2.LINE_4)


          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))



          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

  cv2.imshow("Face emtion", frame)

  if cv2.waitKey(2) & 0xFF == ord('q'):
      break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()