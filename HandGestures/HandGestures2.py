import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the video capture and hand detector
cap = cv2.VideoCapture(0)
detectorHand = HandDetector(maxHands=1)

# Variables to tack the current and last detected gesture
current_gesture = ''
last_gesture = ''

while True:
    _, img = cap.read()

    # Find hands in the image
    hands, img = detectorHand.findHands(img, draw=True) # Set draw=False if you don't want visualization
    if hands:
        lmList = hands[0]['lmList'] # landmarks list
        bbox = hands[0]['bbox'] # Bounding box
        fingers = detectorHand.fingersUp(hands[0])  # Detect which fingers are up

        if bbox:
            # Determine gesture based on finger positions
            if fingers == [1, 1, 1, 1, 1]:
                current_gesture = 'Hi Nova'
            elif fingers == [0, 0, 0, 0, 0]:
                current_gesture = 'Fist'
            elif fingers == [0, 1, 1, 0, 0]:
                current_gesture = 'Victory'
            elif fingers == [1, 0, 0, 0, 0]:
                thumb_fingers_x = lmList[8][0]  # X-coordinate of thumb tip
                wrist_x = lmList[0][0]  # X-coordinate of write
                if wrist_x > thumb_fingers_x:   # Pointing to the right
                    current_gesture = 'Moving Right'
                elif wrist_x < thumb_fingers_x: # Pointing to the left
                    current_gesture = 'Moving Left'
            else:
                current_gesture = ''    # No gesture detected
        print(current_gesture)
    # Show the image
    cv2.imshow('Image', img)
    cv2.waitKey(1)
