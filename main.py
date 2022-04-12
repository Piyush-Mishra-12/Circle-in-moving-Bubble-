import cv2
import numpy as np
import imutils


cap = cv2.VideoCapture("Video 1.mp4")

while True:
    ret, frame = cap.read()
    
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    lg = np.array([30,65,0])
    ug = np.array([60,255,255])

    mask = cv2.inRange(new_frame,lg,ug)

    cordinates = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cordinates = imutils.grab_contours(cordinates)

    for c in cordinates:
        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        else:
            x, y = 0, 0
        new_mask = cv2.circle(mask, (x,y), 10, (0, 0, 255), -1)
        lr = np.array([1,65,0])
        ur = np.array([60,255,255])

        new_new_mask = cv2.inRange(new_mask,lr,ur)
        
        result = cv2.bitwise_and(frame, frame, mask = new_new_mask)
        #result = cv2.circle(result, (x,y), 10, (0, 255, 255), -1)    

        final_result = cv2.bitwise_xor(frame,result)

    cv2.imshow('Frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.imshow('Final result', final_result)


    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destoryAllWindows()
