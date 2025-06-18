import cv2

from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PIDModule import PID

# Load images
background_img = cv2.imread('./Eye-Background.png', cv2.IMREAD_UNCHANGED)       # cv2.IMREAD_UNCHANGED -> # Load the image as-is, including (R,G,B)
iris_img = cv2.imread('./Eye-Ball.png', cv2.IMREAD_UNCHANGED)                   # plus the alpha(transparency) channel, if it exists.
                                                                                # 1st channel: Red, 2nd: Green, 3rd: Blue, 4th: Alpha
                                                                                # image blending: placing the iris on top of the background without a square/rectangular border
# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 480)                 # width. cap.set() is a suggestion, not a gurantee. The actual resolution of my camera is 640(width) x 360(height) <-> 16:9 ratio
cap.set(4, 400)                 # height

# Initialize face detector and PID controller
detector = FaceDetector(minDetectionCon=0.6)
xPID = PID([0.03, 0, 0.06], 640//2, axis=0)      # PID helps track horizontal (X-axis) face position smoothly
                                                        # Since 640 x 360, the center point must be 640//2

# Function to overlay the iris on the background
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
        background[y:y+h, x:x+w, c] = alpha * iris[:,:,c] + (1 - alpha) * background[y:y+h, x:x+w, c]

# Initialize iris position in the center
iris_position = (325, 225)  # (x,y)

cv2.namedWindow('Overlay Result', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Overlay Result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    #print(img.shape)                           # check the actual size: (360, 640, 3)
    img = cv2.flip(img, 1)
    cv2.imshow("Webcam", img)
    img, bboxs = detector.findFaces(img)

    if bboxs:
        cx = bboxs[0]['center'][0]
        resultX = int(xPID.update(cx))
        print(resultX)

        # Update iris position based on resultX
        if resultX > 1:
            iris_position = (400, 225)  # Move iris to the right
        elif resultX < -1:
            iris_position = (250, 225)  # Move iris to the left
        else:
            iris_position = (325, 225)  # Center iris

    print(iris_position)
    # Overlay the iris on the background image
    background_with_iris = background_img.copy()
    overlay_iris(background_with_iris, iris_img, iris_position[0], iris_position[1])

    # Display results and move the window
    cv2.imshow('Overlay Result', background_with_iris)
    cv2.moveWindow('Overlay Result', 0,0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()