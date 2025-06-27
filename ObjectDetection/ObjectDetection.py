import io
import torch        # Imports the Pytorch library, which is used for tensor operations and deep learning (including checking CUDA/GPU availability)
from ultralytics import YOLO
import cv2
import os
import cvzone
from openai import OpenAI
from typing import Literal
import pygame

if torch.cuda.is_available():
    print('CUDA is available.')
    print('GPU Name: ', torch.cuda.get_device_name(0))      # Prints the name of the first GPU
    print('GPU Index: ', torch.cuda.current_device())       # Prints the index (ID) of the currently selected GPU device
    print('Total GPUs: ', torch.cuda.device_count())        # Prints the total number of CUDA-enabled GPUs available

# Set up OpenAI API for Text-to-Speech
# client = OpenAI(api_key='sk-')
client = 'Hello World'

def text_to_speech(text):
    voice: Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] = 'echo'
    response = client.audio.speech.create(model='tts-1', voice=voice, input=text)
    play_audio(response.read())

def play_audio(audio_source):
    pygame.mixer.init()

    if isinstance(audio_source, bytes):
        audio_stream = io.BytesIO(audio_source)
        pygame.mixer.music.load(audio_source)
    elif isinstance(audio_source, str):
        pygame.mixer.music.load(audio_source)

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Load YOLO model
model = YOLO('yolov10n.pt')     # Loads the YOLOv10 nano model

# Define target classes and corresponding images
target_classes = {
    'cell phone': 'Images/cell_phone.jpg',
    'remote': 'Images/remote.jpg',
    'bottle': 'Images/bottle.jpg',
    'scissors': 'Images/scissors.jpg'
}

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not success:
        break

    # Perform inference on images or videos
    # results = model('./Images/image1.jpg', show=True)     # show=True displays the image in a new window with bounding boxes drawn on detected objects.
    # results = model('./Images/people.mp4', show=True)

    # Perform inference on the current frame
    results = model(frame, stream=True)
    ''' Runs the YOLO model on the current frame.
    stream(=True) allows for faster inference with a generator-style output that yields a results object for each frame.
    Otherwise, it returns a result object directly. Generator is like 'loop-ready' box that gives you one result at a time when you iterate over it.
    
      Variable   |       Type         |   Description
    -------------------------------------------------------------------------------------------------------------
    results      |    generator       |   A generator yielding Results objects (one per frame)
    r            |    Results         |   Holds all detections (boxes, class IDs, confidence, etc.) for one frame
    r.boxes      |    Boxes           |   Container holding all individual bounding boxes in that frame
    box          |    Boxes row       |   One detection (e.g., one cell phone or bottle)
    '''

    for r in results:
        for box in r.boxes:
            # Get class ID and confidence
            cls = int(box.cls[0])                       # box.cls -> a PyTorch tensor containing the class index of the detected object
                                                        # Suppose YOLO detects a bottle, and the class ID for 'bottle' is 39. Then box.cls returns tensor([39.])
                                                        # int(box.cls[0]) - int(39.) = 39. YOLO runs on PyTorch, and everything is returned as tensors by default. even if there's just one value.
            class_name = model.names[cls]
            confidence = box.conf[0]

            # Check if detected class is in the target classes and confidence is above 0.3
            if class_name.lower() in target_classes and confidence > 0.3:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                '''box.xyxy: a Pytorch tensor containing the bounding box coordinates of the detected object: [x1, y1, x2, y2]
                x1, y1 = top-left corner of the box. x2, y2 = bottom-right corner of the box
                This is called the 'XYXY format'. and it defines a rectangle on the image.
                Even though this is one box, YOLO keeps the coordinates in a 2D tensor like this:
                box.xyxy -> tensor([[x1,y1,x2,y2]]), shape(box.xyxy) = (1,4), shape(box.xyxy[0])=(4,)
                
                map(function, list) applies the function to every item in the list, one by one
                box.xyxy[0] is something like tensor([150.5, 200.0, 300.9, 400.2)
                map(int, ...) = convert every float to an integer: [150, 200, 300, 400]
                x1, y1, x2, y2 = ... unpacks the 4 numbers 
                '''

                # Draw corner rectangle
                cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), l=30, t=5, rt=0, colorR=(0,255,0))
                # Add label with confidence using cvzone
                cvzone.putTextRect(frame, f'{class_name}{confidence:.2f}',(x1,y1-20))

                text_to_speech(f'I found a {class_name}')
                print(f'Detected: {class_name} with confidence {confidence:.2f}')

                # Display corresponding image for the detected object on the second monitor
                img_path = target_classes[class_name.lower()]
                if os.path.exists(img_path):
                    detected_img = cv2.imread(img_path)


                    # cv2.namedWindow('Detected Image', cv2.WND_PROP_FULLSCREEN)
                    # cv2.setWindowProperty('Detected Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # cv2.moveWindow('Detected Image', 0, 0)
                    cv2.namedWindow('Detected Image', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Detected Image', 600, 400)
                    # cv2.moveWindow('Detected Image', 640, 0)
                    cv2.imshow('Detected Image', detected_img)

    # Display the updated frame with corner rectangles
    cv2.imshow('Camera', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
