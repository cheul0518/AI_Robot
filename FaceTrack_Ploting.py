import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PIDModule import PID
from cvzone.PlotModule import LivePlot

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
X_CENTER = FRAME_WIDTH // 2
SERVO_MAX = 180
SERVO_MIN = 0
INITIAL_X_ANGLE = 105

# Initialize components
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

detector = FaceDetector(minDetectionCon=0.75)
xPID = PID([0.015, 0, 0.06], X_CENTER, axis=0)
xPlot = LivePlot(yLimit=[0, FRAME_WIDTH], char='X')         # LivePlot is a class in the cvzone library designged to visualize changing values over time.
                                                            # It's used to plot the horizontal (X-axis) position of the detected face center in each frame.
                                                            # yLimit [min,max] sets the vertical axis range for plotting the X-coordinate values over time.
                                                            # It initializes an empty list of values. As you call .update(value), it appends each new value to the list.
xAngle = INITIAL_X_ANGLE

while True:
    success, img = cap.read()
    if not success:
        print('Failed to read from the camera.')
        break

    img, bboxs = detector.findFaces(img)                    # bboxs is a list of dictionaries. bbox is like {'id':0,'bbox':[x1,y1,w1,h1],'score':[confidence],'center':[cx1,cy1]}.

    if bboxs:
        x, y, w, h = bboxs[0]['bbox']
        cx, cy = bboxs[0]['center']
        print(cx)
        resultX = int(xPID.update(cx))                      # xPID.update(cx) computes how far off-center the face is.

        xAngle = max(SERVO_MIN, min(SERVO_MAX, xAngle + resultX))       # Updates xAngle to simulate turning the servo toward the face. It's clamped between 0-180 degrees.

        imgPlotX = xPlot.update(cx)                         # xPlot starts as an empty list of values. As you call .update(value), it appends each new value to the list.

        img = xPID.draw(img, [cx,cy])                       # Draws a vertical line at the target position(320), and another line from the current position to the target

        cv2.imshow('Image X plot', imgPlotX)
    else:
        print('No face detected')

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

