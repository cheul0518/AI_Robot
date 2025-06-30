""" Overview
1. Detects a face using a webcam.
2. Determines the horizontal position of the face.
3. Uses a PID controller to calculate how much to move a cartoon eye (iris image).
4. Overlays the iris on an eye background image.
5. Simulates tracking the face by making the iris "look" left, right, or center.
Eventually, this can control a servo motor using an Arduino or Raspberry Pi — but that part is commented out.
"""

# Libraries
import cv2                                                                      # OpenCV, used for image processing and webcam interaction
from cvzone.FaceDetectionModule import FaceDetector                             # From cvzone, wraps around a face detection model (uses Mediapipe internally)
from cvzone.PIDModule import PID                                                # A simple Proportional-Integral-Derivative controller to calculate smooth changes
""" Arduino component - to be replaced by a Raspberry Pi in future implementations.
from cvzone.SerialModule import SerialObject
"""

# Load images
background_img = cv2.imread('Images/Eye-Background.png', cv2.IMREAD_UNCHANGED)       # cv2.imread(): Load an image from a file. Returns a Numpy array
                                                                                # cv2.IMREAD_COLOR(1): default. no alpha channel. 3 channels:BGR
iris_img = cv2.imread('Images/Eye-Ball.png', cv2.IMREAD_UNCHANGED)                   # cv2.IMREAD_GRAYSCALE(0): load the image in grayscale. 1 channel
                                                                                # cv2.IMREAD_UNCHANGED(-1): load the image as-is, including the alpha channel(transparency). 4 channels:BGRA

# Initialize camera
cap = cv2.VideoCapture(0)                                                       # Initializes the webcam (0 is the default camera).
if not cap.isOpened():                                                          # Exits if the webcam fails to open.
    exit('Cannot open camera.')

# Initialize face detector and PID controller
detector = FaceDetector(minDetectionCon=0.6)                                    # Initializes a face detector that filters out weak directions
xPID = PID([0.03, 0, 0.06], 640//2, axis=0)                              # P: 0.03, I: 0, D: 0.06, 640//2: Target x-position, axis=0: x axis
                                                                                # Returns a value that tells how far and in which direction the current face is from the center.
                                                                                # Logitech webcam: 640 * 480,  iMacCam: 1920 * 1080

""" Arduino component - to be replaced by a Raspberry Pi in future implementations.
# Initialize the starting angle of a servo motor.
xAngle = 105
# Initialize Arduino serial communication
arduino = SerialObject(digits=3)
# Define the threshold value for the center
center_threshold = 2    # Adjust this value as needed
"""

""" 
Puts the iris onto the background image at position (x, y).
Uses alpha blending (alpha = iris[:, :, 3]/255.0) to mix transparent pixels correctly.
Ensures the iris doesn’t go outside the bounds of the background image.
"""
def overlay_iris(background, iris, x, y):
    h, w = iris.shape[:2]
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        iris = iris[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        iris = iris[:h]
    alpha = iris[:, :, 3]/255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = alpha * iris[:, :, c] + (1-alpha) * background[y:y+h, x:x+w, c]

def main():
    # Create a named window and set it to full screen
    cv2.namedWindow('Overlay Result', cv2.WND_PROP_FULLSCREEN)              # cv2.WND_PROP_FULLSCREEN mode used with cv2.setWindowProperty() for full screen mode
    cv2.setWindowProperty('Overlay Result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    """ Why? cv2.namedWindow is necessary.
    You must create the window before you try to set its properties.
    cv2.setWindowProperty() can only modify an already existing window.
    If you skip cv2.namedWindow(), the window is automatically created when cv2.imshow() is called.
    But here's the catch: auto-created windows use default settings (e.g., cv2.WINDOW_AUTOSIZE) — and by then it's too late to change full screen or resizable mode.
    """

    # Initialize iris position in the center
    iris_position = (325, 225)

    while True:
        success, img = cap. read()              # Capture frame by frame
                                                # success is true if the frame was read successfully. img is the actual image (frame) data
        if not success:                         # exit if success is false
            print('Failed to grab frame.')
            break

        img = cv2.flip(img, 1)          # flipCode
                                                #  0 : Vertical: Flips image up-down (over x-axis)
                                                #  1 : Horizontal: Flips image left-right (over y-axis)
                                                # -1 : Horizontal and vertical: over x-axis and y-axis

        img, bboxs = detector.findFaces(img)    # findFaces returns "the image with annotations (like rectangles)", and "a list of bounding boxes" where faces are found.

        if bboxs:                               # In Python, an empty list evaluates to False
            cx = bboxs[0]['center'][0]          # bboxs is a list of dictionaries. bbox is like {'id':0,'bbox':[x1,y1,w1,h1],'score':[confidence],'center':[cx1,cy1]}

            # Calculate the x-angle adjustment based on PID
            resultX = int(xPID.update(cx))      # xPID.update(cx): calculates how far the face is from center(320) and returns a correction value
            # Update iris position based on resultX
            if resultX > 1:
                iris_position = (400, 225)      # Move iris to the right
            elif resultX < -1:
                iris_position = (250, 225)      # Move iris to the left
            else:                               # iris at center
                iris_position = (325, 225)

            """ Arduino component - to be replaced by a Raspberry Pi in future implementations.
            # Control the servo based on xAngle
            if abs(resultX) > center_threshold:
                xAngle += resultX
            arduino.sendData([180, 0, xAngle])
            """
        else:
            print('No face detected.')

        # Overlay the iris on the background image
        background_with_iris = background_img.copy()
        overlay_iris(background_with_iris, iris_img, iris_position[0], iris_position[1])

        cv2.imshow('img', img)                                  # Display the resulting frame in a window titled "Image"
        cv2.imshow('Overlay Result', background_with_iris)
        cv2.moveWindow('Overlay Result', 0, 0)            # cv2.moveWindow(window_name, x, y). It moves an OpenCV window to a specific position on your screen.
                                                                        # window_name: Name of the window. It must match what you passed to cv2.imshow()
                                                                        # x: Distance in pixels from the left edge of screen. y: Distance in pixels from the top edge of screen.

        if cv2.waitKey(1) & 0xFF == ord('q'):                           # cv2.waitKey(1) waits 1 millisecond for a key press
            break                                                       # & 0xFF: keep only the lowest 8 bits (ASCII) b/c cv2.waitkey() sometimes returns a value with extra bits
                                                                        # key_raw  = 0x100071 -> Binary: 0001 0000 0000 0000 0111 0001
                                                                        # 0xFF     = 0x0000FF -> Binary: 0000 0000 0000 0000 1111 1111
                                                                        # Filtered =                     0000 0000 0000 0000 0111 0001 -> 0x71 = 113 = 'q'
                                                                        # q key breaks the loop and ends the program

    cap.release()                                                       # Releases the camera
    cv2.destroyAllWindows()                                             # Closes any OpenCV windows (e.g. the "Image" window.)

if __name__=='__main__':
    main()