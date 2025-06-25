import pickle
import cv2
import face_recognition
import numpy as np
import pygame
from openai import OpenAI
from typing import Literal
import io
import os

# Set up Open API for Text-To-Speech
#client = OpenAI(api_key='sk-')
client = 'Hello World'

def text_to_speech(text):
    ''' Converts text to speech using OpenAI's Text-to-Speech API
    :param text: text to convert to speech
    '''
    voice: Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] = 'echo'    # name of voice types supported by OpenAI's Text-to-Speech
    response = client.audio.speech.create(model='tts-1', voice=voice, input=text)
    play_audio(response.read())

def play_audio(audio_source):
    ''' Plays audio content using pygame. Supports both binary audio content
    (e.g., from text-to-speech APIs) and file path to an .mp3 file.
    :param audio_source: binary audio content or a file path to an .mp3 file
    '''
    pygame.mixer.init()

    if isinstance(audio_source, bytes): # Binary audio content
        audio_stream = io.BytesIO(audio_source)
        pygame.mixer.music.load(audio_stream)
    elif isinstance(audio_source, str): # File path
        pygame.mixer.music.load(audio_source)

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def main():
    print('Loading Encode File ...')
    with open('EncodeFile.p', 'rb') as f:           # Loads the file named 'EncodeFile.p which contains a list of face encodings and corresponding personIDs
        encodeListKnownIDs = pickle.load(f)

    encodeListKnown, personIDs = encodeListKnownIDs
    print('Encode File Loaded')

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    recognized_name = ''

    # Initialize video
    video_path = './Resources/casual Eyes.mp4'
    vid = cv2.VideoCapture(video_path)

    while True:

        # Video
        if not vid.isOpened():
            print(f'Error: Could not open video file at {video_path}')
            break

        ret, frame = vid.read()

        # Greeting and Camera
        success, img = cap.read()
        if not success:
            print('Failed to capture image')
            break

        img = cv2.flip(img, 1)
        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)         # dsize: desired size in pixels (width, height). If dsize's provided, fx and fy will be ignored.
                                                                                # dst: allows OpenCV to write into an existing image buffer. Normally ignored (None)
                                                                                # fx, fy: scale factors along width and height
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)                    # Finds all faces in the image. It returns a list of tuples [(top,right,bottom,left),...]
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)    # Returns a list of numpy.ndarray

        recognized_name = ''

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            # face_recognition.compare_faces(known_encodings, face_to_check, tolerance=0.6)
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)   # Compare a new face encoding (encodeFace) with a list of known face encodings (encodeListKnown)
                                                                                    # Returns a list of boolean values - True / False
                                                                                    # Uses Euclidean distance internally and compares against a threshold tolerance)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)   # Calculates the Euclidean distance between face_to_check and each of the known_encodings
                                                                                    # Returns a list of floats (# of known_encodings) - the smaller the distance, the more similar the face


            matchIndex = np.argmin(faceDis)         # Returns the index of the smallest value in a NumPy array
                                                    # e.g., faceDis = [0.52, 0.66, 0.29, 0.4], then matchIndex is 2 b/c faceDis[2] = 0.29 is the smallest

            if matches[matchIndex]:                         # Why use both .compare_faces and .face_distance?
                                                            # .compare_faces() ensures it's close enough to be trusted -> is the best match close enough?
                                                            # .face_didstance() ensures you pick the closest known face -> who is the most similar face?
                personID = personIDs[matchIndex]
                recognized_name = personID
                print(f'Detected Face ID: {personID}')

                # Scale the face locations back to the original image size
                top,right,bottom,left = faceLoc
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a rectangle around the detected face
                cv2.rectangle(img, (left,top), (right,bottom), (0,255,0),2)

                # Optionally, add a label with the person ID
                cv2.putText(img, personID, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
            else:
                recognized_name = 'Stranger'

        if recognized_name:
            if recognized_name == 'Stranger':
                greeting_text = 'Hello Stranger'
                text_to_speech('Hello Stranger')
            else:
                greeting_text = f'Hello {recognized_name}!'
                text_to_speech(f'Hello {recognized_name}')

            greeting_img = np.zeros((200, 640, 3), np.uint8)
            cv2.putText(greeting_img, greeting_text, (110, 100), cv2.QT_FONT_NORMAL, 2, (255, 255, 255), 2)

            cv2.namedWindow('Greeting', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Greeting', 640, 200)
            cv2.moveWindow('Greeting', 0, 0)
            cv2.imshow('Greeting', greeting_img)

        cv2.imshow('Cam', img)

        cv2.namedWindow('Vid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vid', 600, 400)
        cv2.moveWindow('Vid', 640, 0)

        if ret:
            cv2.imshow('Vid', frame)
        else:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()