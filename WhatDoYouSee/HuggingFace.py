import io, cv2, pygame              # "io" for handling in-memory byte streams; "cv2" for webcam and video processing(OpenCV); "pygame" for audio playback
from openai import OpenAI           # to connect with OpenAI's API for text-to-speech
from typing import Literal
from PIL import Image               # "PIL" stands for Python Imaging Library, for image loading
from transformers import BlipProcessor, BlipForConditionalGeneration        # "transformers" for using the BLIP model (image captioning)

#client = OpenAI(api_key='sk')      # You create an OpenAI client with an API key - this is used for text-to-speech
client = 'Temporary key'


''' [processor & model]

<processor>: a helper that prepares the raw image (and text, if needed) into the correct input format for the model
    - Loads the image and resizes/normalizes it.
    - Convert it to tensors (PyTorch tensors).
    - Handles tokenization if there's text input too.
    - Handles decoding of model outputs back to text.
    
<model>: the neural network that does the actual prediction
    - Takes the processed tensors.
    - Runs them through its layers.
    - Generates a prediction(a caption for the image).
'''
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')



def text_to_speech(text):
    voice: Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] = 'echo'        # name of voice types supported by OpenAI's Text-to-Speech
    response = client.audio.speech.create(model='tts-1', voice=voice, input=text)
    play_audio(response.read())

def play_audio(audio_source):
    pygame.mixer.init()     # Initializes the pygame.mixer, which handles audio playback.

    if isinstance(audio_source, bytes):     # Checks if audio_source is a bytes object (i.e., raw audio data (TTS))
        audio_stream = io.BytesIO(audio_source)     # Wraps the binary audio content in a Bytes IO object to make it act like a file-like stream
        pygame.mixer.music.load(audio_stream)       # Loads the audio from the BytesIO stream into pygame's music player
    elif isinstance(audio_source, str):     # File path
        pygame.mixer.music.load(audio_source)

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():    # Enters a loop that continues as long as the music is still playing
        pygame.time.Clock().tick(10)        # Limits the loop to run at most 10 times per second (i.e., every 100ms).
                                            # Only 10 loops happens in one second
                                            # 10 FPS means one loop every 100 milliseconds (because 1 second / 10 = 0.1 sec = 100 ms
    ''' Why this is useful?
    Without tick(), loop could run as fast as your computer allows, using 100% CPU and maybe causing problems
    (like too many frames, inconsistent speeds, or audio issues)
    So tick() keeps your game or loop running at a steady, predictable speed 
    '''

def query(filename):
    '''
    :param: img
    :return: sentence
    1. Open an image file
    2. Preprocess it for the BLIP model
    3. Run it through the model to generate a text sequence
    4. Decode that sequence back to a plain sentence
    5. Return that sentence
    '''
    raw_image = Image.open(filename).convert('RGB')     # Guarantees that the image is in the expected format: 3 channels, no alpha
    inputs = processor(raw_image, return_tensors='pt')  # pt is Pytorch
    ''' processor(...)
    - Resizes the image as needed
    - Normalizes pixel values (to match what the model was trained on)
    - Converts the image to a PyTorch tensor (because of return_tensors='pt')
    - Creates a dictionary with the tensors BLIP expects
    {'pixel_values' : tensor of shape [batch size, channel, height, width],
     'input_ids' : (if needed for text input) 
     ...
    } '''
    out = model.generate(**inputs)      # **inputs unpacks the dictionary from the processor, passing pixel_values (and other args if needed) into the model
                                        # The model uses its vision encoder to process the image and its language decoder to generate a caption
                                        # out is a tensor of token IDs - [225, 192, 1028, 2011, 123] - which is not human-redable yet
    return processor.decode(out[0], skip_special_tokens=True)

def main():


    # Initializes webcam for real-time image capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit('Error: Could not open webcam.')

    # Play back a video file
    video_path = './Videos/casual Eyes.mp4'
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        exit(f'Error: Could not open video file at {video_path}.')

    # Path to the photo click sound effect
    audio_path = './Audio/photo-click.mp3'

    print('Press "s" to capture an image or "q" to quit.')

    while True:

        ret, frame = vid.read()         # Read the next video frame
        success, img = cap.read()       # Capture a frame from the webcam
        if not success:
            print('Failed to capture image.')
            break

        img = cv2.flip(img, 1)      # Flip the image horizontally for a mirror effect
        cv2.imshow('Webcam', img)

        # Show video frame or loop the video if it ends
        if ret:
            cv2.imshow('Vid', frame)
        else:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        key = cv2.waitKey(1)        # Wait for key press

        if key == ord('s'):
            cv2.imwrite('image.jpg', img)       # Save the captured image
            play_audio(audio_path)
            print('Analyzing the captured image...')

            # Processing on the frame
            output = query('image.jpg')
            print(output)
            text_to_speech(f'Yes, I see, {output}')     # Speak out the result
        elif key == ord('q'):
            print('Exiting')
            break

    # Release the resources
    cap.release()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()