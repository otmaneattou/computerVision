"""tuto Advanced computer vision youtube part 1 hand tracking bases  
lien : https://www.youtube.com/watch?v=01sAkU_NvOY&t=4105s
31 land marks 
"""
import cv2 
import mediapipe as mp 
import time

from numpy import result_type
from numpy.core.fromnumeric import ptp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

pTime = 0# pri
cTime = 0#current time



while True:
    Success , img = cap.read()
    imgRGB =  cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)# processing 
    
    if results.multi_hand_landmarks :
        for handLms in results.multi_hand_landmarks :
            for id ,lm in enumerate(handLms.landmark):
                # print(id,lm)
                h ,w ,c = img.shape
                cx ,cy = int(lm.x*w) , int(lm.y*h)
                # print(id,cx,cy)
                # if id== 0 :
                #     cv2.circle(img,(cx,cy) ,15,(255,0,255,cv2.FILLED)) # plot land mark 0 
                
                
                    
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS) # print land marks plus connection  
            
    #calculer FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # plot 
    cv2.putText(img , str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
