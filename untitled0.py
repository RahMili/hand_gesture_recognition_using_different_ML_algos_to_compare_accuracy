import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import imutils
from directkeys import PressKey, ReleaseKey, W, A, S, D

PressKey(W)

img_array = np.empty((1, 200*80))
img_label = np.empty((1))
#directory = "D:/Machine Learning A-Z™ Hands-On Python & R In Data Science/practice/"
CATEGORIES = ["up", "down", "left", "right"]
for category in CATEGORIES:  
    path = category  
    for img in os.listdir(path):  
        im = os.path.join(path, img)
        image = cv2.imread(im)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 80))
        gray = np.reshape(gray, (1, 200*80))
        img_array = np.append(img_array, gray, axis = 0)
        img_label = np.append(img_label, CATEGORIES.index(category))

img_array = np.delete(img_array, (0), axis=0)
img_label = np.delete(img_label, (0), axis=0)

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(img_array, img_label, test_size = 0.1, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(train_X, train_Y)

# Predicting the Test set results
y_pred = classifier.predict(test_X)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y, y_pred)
print(cm)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh_delta = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=1)
    contours, hierarchy  = cv2.findContours(thresh_delta, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        i=0
        if cv2.contourArea(cnt) > 7500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            cropped = thresh_delta[y:y + h, x:x + w] #cropping region of interest i.e. face area from  image
            cropped2=cv2.resize(cropped,(200,80))
            Y_img = np.reshape(cropped2, (1, 200*80))
            Y_pred = classifier.predict(Y_img)
            print(Y_pred)
            print(int(Y_pred[i]))
            cv2.putText(frame, CATEGORIES[int(Y_pred[i])], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            '''if CATEGORIES[int(Y_pred[i])] == CATEGORIES[0]:
                PressKey(W)
                ReleaseKey(W)
            elif CATEGORIES[int(Y_pred[i])] == CATEGORIES[1]:
                PressKey(S)
                ReleaseKey(S)
            elif CATEGORIES[int(Y_pred[i])] == CATEGORIES[2]:
                PressKey(A)
                ReleaseKey(A)
            elsif CATEGORIES[int(Y_pred[i])] == CATEGORIES[3]:
                PressKey(D)
                ReleaseKey(D) ''' 
            i = i + 1
    cv2.imshow('frame', frame)
    cv2.imshow('thresh', thresh_delta)

    k = cv2.waitKey(1)
    
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
