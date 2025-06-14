# This is an example of code using PID (Proportional-Integral-Derivative) control.

import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PIDModule import PID

cap = cv2.VideoCapture(0)                                   # Setup the webcam (default camera = 0)

cap.set(3, 640)                                 # Frame Width
cap.set(4, 480)                                 # Frame Height

detector = FaceDetector(minDetectionCon=0.6)                # Only detects faces if confidence >= 0.6
xPID = PID([0.015,0,0.06],640//2,axis=0)             # Initialize a PID controller for the horizontal axis (x-axis)
                                                            # P=0.015, I=0, D=0.06
                                                            # 320(640//2) is the target value. /: floating point divison //: integer division
                                                            # axis = 0 : horizontal axis -> it draws error line based on x position(cx)
                                                            # axis = 1 : vertical axis -> it would draw based on y position(cy)
# yPID = PID([0.015,0,0.06],480//2,axis=1)

while True:
    success, img = cap.read()

    if not success:
        print("Failed to grab frame.")
        break

    img, bboxs = detector.findFaces(img)                    # automatically draws a rectangle around each detected face on the img it returns

    if bboxs:
        x, y, w, h = bboxs[0]['bbox']                       # bboxs is a list of dictionaries. bbox[0] is like {'id':0,'bbox':[x1,y1,w1,h1],'score':[confidence],'center':[cx1,cy1]}
        cx, cy = bboxs[0]['center']
        # Calculate the x-angle adjustment based on PID
        resultX = int(xPID.update(cx))                      # xPID.update(cx): calculates how far the face is from center(320) and returns a correction value
        print(resultX)                                      # P: how far off are we? , I: how long have we been off? , D: How fast is the error changing?
        # resultY = int(yPID.update(cy))
        # print(resultY)

        # Draw the PID information on the image
        img = xPID.draw(img, [cx,cy])                  # Draws a line at the target position(320), and a line from the current position to the target
        # img = yPID.draw(img, [cx, cy])

    cv2.imshow("Image", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()