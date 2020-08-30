import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(1)
img_counter = 0
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh_delta = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=1)
    cnts = cv2.findContours(thresh_delta, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE)
    lower_green = np.array([0, 0, 0])
    upper_green = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cnts = imutils.grab_contours(cnts)
    for contour in cnts:
        if cv2.contourArea(contour) > 7500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
            cropped = thresh_delta[y:y + h, x:x + w]
    cv2.imshow('frame', frame)
    cv2.imshow('thresh', thresh_delta)
    #cv2.imshow('deltaframe', delta_frame)
       
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpeg".format(img_counter)
        cv2.imwrite(img_name, cropped)
        print("{} written!".format(img_name))
        img_counter = img_counter + 1
    
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
    