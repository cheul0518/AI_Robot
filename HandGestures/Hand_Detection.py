import cv2
from cvzone.HandTrackingModule import HandDetector  # cvzone is a wrapper around OpenCV and MediaPipe  that makes hand and face tracking easier

cap = cv2.VideoCapture(0)
detectorHand = HandDetector(maxHands=1) # The limit of maxHands is 2

while True:
    _, img = cap.read()     # This reads a single frame from the webcam

    hands, img = detectorHand.findHands(img, draw=True)   # Change to draw=False if you dont want visualization
    '''
    hands -> a list of detected hands (each with landmarks, bounding box, center, etc.)
    {'lmList': [...],           # Landmarks list (21 points, each with x,y,z)
    'bbox': [...],              # Bounding box [x, y, w, h]
    'center': [...],            # Center of hand [x, y]
    'type': "Left" or "Right"   # Which hand it is
    }
    img -> the same image but with hand annotations drawn on it
    
        '''
    if hands:
        lmList = hands[0]['lmList'] # Landmakrs List
        bbox = hands[0]['bbox']     # Bounding box
        fingers = detectorHand.fingersUp(hands[0])  # Detect which fingers up

    cv2.imshow('Image', img)
    cv2.waitKey(1)
