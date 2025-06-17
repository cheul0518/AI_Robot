import cv2

cap = cv2.VideoCapture(0)                       # VideoCapture object: camera index 0 - 5

cap.set(3, 640)                     # 3: width  640 pixels
cap.set(4, 480)                     # 4: height 480 pixels

if not cap.isOpened():                          # check if the camera opened successfully
    print('Cannot open camera')
    exit()

while True:
    success, img = cap.read()                   # Capture frame by frame
                                                # success is true if the frame was read successfully
                                                # img is the actual image (frame) data

    if not success:                             # exit if success is False
        print('Failed to grab frame.')
        break

    cv2.imshow('Image', img)            # Display the resulting frame in a window titled "Image"
                                                # Updates on every loop cycle to create a live video

    if cv2.waitKey(1) & 0xFF == ord('q'):       # cv2.waitkey(1) waits 1 millisecond for a key press
        break                                   # & 0xFF: keep only the lowest 8 bits (ASCII). cv2.waitkey() sometimes returns a value with extra bits
                                                # key_raw  = 0x100071 -> Binary: 0001 0000 0000 0000 0111 0001
                                                # 0xFF     = 0x0000FF -> Binary: 0000 0000 0000 0000 1111 1111
                                                # Filtered =                     0000 0000 0000 0000 0111 0001 -> 0x71 = 113 = 'q'
                                                # q key breaks the loop and ends the program

cap.release()                                   # Releases the camera
cv2.destroyAllWindows()                         # Closes any OpenCV windows (e.g. the "Image" window.)