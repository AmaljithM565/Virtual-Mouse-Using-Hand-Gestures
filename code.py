import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

pyautogui.FAILSAFE=False

cap=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=1)

screen_w, screen_h=pyautogui.size()

click_threshold=40
clicked=False

while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    h,w,_=frame.shape

    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            index_tip=hand.landmark[8]
            thumb_tip=hand.landmark[4]

            ix,iy=int(index_tip.x*w), int(index_tip.y*h)
            tx,ty=int(thumb_tip.x*w), int(thumb_tip.y*h)

            screen_x=np.interp(ix,(0,w),(0,screen_w))
            screen_y=np.interp(iy,(0,h),(0,screen_h))

            pyautogui.moveTo(screen_x, screen_y)

            distance=math.hypot(tx-ix, ty-iy)

            if distance < click_threshold and not clicked:
                pyautogui.click()
                clicked=True

            if distance > click_threshold+10:
                clicked=False    

            cv2.circle(frame, (ix,iy), 10, (0,255,0), -1)
            cv2.circle(frame, (tx,ty), 10, (0,0,255), -1)

    cv2.imshow("Mouse",frame)

    if cv2.waitKey(1)& 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows    

