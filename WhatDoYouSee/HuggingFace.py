import io
import os
import cv2
import pygame
from openai import OpenAI
from typing import Literal
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

#client = OpenAI(api_key='sk')
client = 'Hello World'

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')



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

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def query(filename):
    raw_image = Image.open(filename).convert('RGB')
    inputs = processor(raw_image, return_tensors='pt')
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_toekns=True)

def main():
    cap = cv2.VideoCapture(0)
    print('Press "s" to capture an image or "q" to quit.')

    audio_path = './Audio/photo-click.mp3'

    video_path = './Videos/casual Eyes.mp4'
    vid = cv2.VideoCapture(video_path)


    while True:

        if not vid.isOpened():
            print(f'Error: Could not open video file at {video_path}.')
            break

        ret, frame = vid.read()

        success, img = cap.read()
        if not success:
            print('Failed to capture image.')
            break

        img = cv2.flip(img, 1)
        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1)

        if ret:
            cv2.imshow('Vid', frame)
        else:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)


        if key == ord('s'):
            cv2.imwrite('image.jpg', img)
            play_audio(audio_path)
            print('Analyzing the captured image...')

            # Processing on the frame
            output = query('image.jpg')
            print(output)
            text_to_speech(f'Yes, I see, {output}')
        elif key == ord('q'):
            print('Exiting')
            break


    cap.release()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()