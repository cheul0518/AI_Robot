import io

import cv2, pygame
from cvzone.HandTrackingModule import HandDetector
from openai import OpenAI
from typing import Literal

# client = OpenAI(api_key='sk')
client = 'Temporary Key'

def text_to_speech(text):
    voice: Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] = 'echo'
    response = client.audio.speech.create(model='tts-1', voice=voice, input=text)
    play_audio(response.read())

def play_audio(audio_source):
    pygame.mixer.init()

    if isinstance(audio_source, bytes):
        audio_stream = io.BytesIO(audio_source)
        pygame.mixer.music.load(audio_stream)
    elif isinstance(audio_source, str):
        pygame.mixer.music.load(audio_source)

def main():

    # Play back a video file
    video_path = './Videos/casual Eyes.mp4'
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        exit(f'Error: could not open video file at {video_path}')

    # Path to the photo click sound effect
    audio_path = './Audio/photo-click.mp3'

    # Initialize the video capture and hand detector
    cap = cv2.VideoCapture(0)
    detectorHand = HandDetector(maxHands=1)

    current_gesture = ""

    while True:
        ret, frame = vid.read()     # Read the next video frame
        success, img = cap.read()   # Capture a frame from the webcam
        if not success:
            print('Failed to capture image.')
            break

        img = cv2.flip(img, 1)  # Flip the image horizontally for a mirror effect
        # Find hands in the image
        hands, img = detectorHand.findHands(img, draw=False)     # Set draw=False if you don't want visualization

        if ret:
            cv2.imshow('Vid', frame)
        else:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if hands:
            lmList = hands[0]['lmList']     # Landmarks list
            bbox = hands[0]['bbox']         # Bounding box
            fingers = detectorHand.fingersUp(hands[0])      # Detect which fingers are up

            # Flip the label to match mirror view
            hand = hands[0]
            corrected_label = "Right" if hand['type'] == 'Left' else "Left"

            if bbox:
                x, y, w, h = bbox
                cv2.putText(img, corrected_label, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 0, 255), 3)

                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = w + 2 * padding
                h = h + 2 * padding
                cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)

                # Determine gesture based on finger positions
                if fingers == [1, 1, 1, 1, 1]:
                    current_gesture = 'Hi, there'
                    # text_to_speech('Hello')
                elif fingers == [0, 0, 0, 0, 0]:
                    current_gesture = 'Fist'
                    # text_to_speech('Bye')
                elif fingers == [0, 1, 1, 0, 0]:
                    current_gesture = 'Victory'
                    # play_audio(audio_path)
                    # text_to_speech('Capture your selfie')
                    cv2.imwrite("image.jpg", img)
                elif fingers == [1,0,0,0,0]:
                    thumb_finger_x = lmList[8][0]       # X coordinate of thumb tip
                    wrist_x = lmList[0][0]
                    if wrist_x > thumb_finger_x:
                        current_gesture = 'Moving Left'
                        # text_to_speech("What's on the left side")
                    elif wrist_x < thumb_finger_x:
                        current_gesture = 'Moving Right'
                        # text_to_speech("What's on the right side")
                else:
                    current_gesture = ""

        print(current_gesture)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            print('Exiting')
            break

    cap.release()
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()