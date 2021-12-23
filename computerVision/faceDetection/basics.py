import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

# Data to process
cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgRGB)
    h, w, c = img.shape
    if result.detections:
        # mpDraw.draw_landmarks(img, result.pose_landmarks,
        #                      mpPose.POSE_CONNECTIONS)
        for id, detection in enumerate(result.detections):
            # mpDraw.draw_detection(img, detection)
            #cx, cy = int(lm.x*w), int(lm.y*h)
            #cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h),\
                int(bboxC.width*w), int(bboxC.height*h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cTime = time.time()  # current time
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
