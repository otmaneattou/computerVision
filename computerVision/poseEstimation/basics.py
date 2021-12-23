import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Data to process
cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    h, w, c = img.shape
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cTime = time.time()  # current time
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
