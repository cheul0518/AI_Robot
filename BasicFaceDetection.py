import cv2
import cvzone                                               # pip install cvzone, pip install mediapipe
from cvzone.FaceDetectionModule import FaceDetector         # A ready-to-use face detection class from cvzone that internally uses MediaPipe for detection

# Initialize components
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = FaceDetector(minDetectionCon=0.75)                   # This creates a face detector
                                                                # Minimum detection confidence threshold (argument)
while True:
    success, img = cap.read()                                   # success is true if the frame was read successfully
                                                                # img is the actual image (frame) data

    if not success:                                             # exit if success is false
        print('Failed to grab frame.')
        break

    img, bboxs = detector.findFaces(img)                        # findFaces returns "the image with annotations (like rectangles)", and "a list of bounding boxes" where faces are found.

    if bboxs:                                                   # In Python, an empty list evaluates to False
        for bbox in bboxs:                                      # bboxs is a list of dictionaries. bbox is like {'id':0,'bbox':[x1,y1,w1,h1],'score':[confidence],'center':[cx1,cy1]}
            x, y, w, h = bbox['bbox']                           # x,y: top-left corner of the rectangle(position), w: width of the face box, h: height of the face box
            # Stylized rectangle
            cvzone.cornerRect(img,(x,y,w,h),l=10)          # img: the current image, (x,y,w,h): bouding box, l= length of the corner lines
    else:
        print('No face detected')

    # Display the video feed
    cv2.imshow('Face Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()