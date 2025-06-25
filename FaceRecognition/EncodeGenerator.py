import os
import pickle                       # Serializing and deserializing Python objects
                                    # Serialization: convert a Python object (like a list, dictionary, or model) into a byte stream so it can be saved to a file or sent over a network
                                    # Deserialization: convert that byte stream back into the original Python ojbect.
                                    # Think of pickle as a way to save and load Python data

import cv2
import face_recognition             # Python wrapper around dlib's state-of-the-art face recognition algorithms.
                                    # 1) Detect faces in images, 2) Find landmarks (eyes, nose, mouth), 3) Encode faces into numerical vectors, 4) Compare faces to see if they match

# Importing images
folderPath = 'Images'
pathList = os.listdir(folderPath)   # Lists all filenames in the 'Images' folder, e.g., ['Elon.png', 'Erica.png', ...]
print(pathList)
imgList = []
personIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))  # os.path.join(...): constructs the full path to the image file.
    personIds.append(os.path.splitext(path)[0])                 # os.path.splitext(path)[0]: removes the file extension, e.g., Elon.png -> Elon
                                                                # os.path.splitext(path)[1]: the file extension, e.g., Elon.png -> .png
    fileName = f'{folderPath}/{path}'

print(personIds)

# Given a list of images, returns a list of their 128 dimensional face encodings
def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # converts BGR(OpenCV default) to RGB (what face_recognition expects)
        '''
        # Get all face encodings, and then check how many faces were found
        encodings = face_recognition.face_encodings(img)
        print(f"{len(encodings)} face(s) found in the image.")
        '''
        encode = face_recognition.face_encodings(img)[0]        # .face_encodings() extracts the face encoding (128-dimensional vectors) from an image
                                                                # This assumes each image contatins exactly one face. If no face's found, an index error'll be raised.
                                                                # Each encoding represents the unique features of a face in a way that allows for comparison and recognition
                                                                # The 128 values are like a fingerprint - useful for comparison, not reconstruction (learned by a neural network to differentiate faces, not to recreate them)
                                                                # [0] -> the first face's 128D encoding, [1] -> the second face's 128D encoding, and so on...
        encode_list.append(encode)

    return encode_list

print('Encoding Started...')
encodeListKnown = find_encodings(imgList)
encodeListKnownWithIds = [encodeListKnown, personIds]
print('Encoding Complete')

file = open('EncodeFile.p', 'wb')
pickle.dump(encodeListKnownWithIds, file)                       # pickle.dump(obj, file): take the Python object "obj" and write it into the file "file" in a special binary format.
                                                                # with open("EncodeFile.p", "rb") as f: # load it back
                                                                #   loaded_data = pickle.load(f)
file.close()
print('File Saved')