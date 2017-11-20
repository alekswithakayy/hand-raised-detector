import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import face_recognition
import threading
import shutil
import os
import cv2

from PIL import Image
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from collections import namedtuple
from io import StringIO
from collections import namedtuple

# Set up connection to Clarifai API
app = ClarifaiApp(api_key='a41852077a72486398e063fb60936b65')
hand_raised_detection = app.models.get('hand-raised-detector')

# Extract individual people out of image and return an array of clarifai images
def extract_people(faces, image):

    people = []

    for index,face in enumerate(faces):
        image_height, image_width, _ = image.shape

        face_top, face_right, face_bottom, face_left = face
        # Calculate width and height of face
        face_height = face_bottom - face_top
        face_width = face_right - face_left

        # Roughly outline person's body using face making sure to stay within bounds of the image
        person_top = max((face_top - face_height * 4), 0)
        person_right = min((face_right + int(face_width * 3.5)), image_width)
        person_bottom = min((face_bottom + int(face_height * 3.5)), image_height)
        person_left = max((face_left - int(face_width * 1.5)), 0)

        # Extract person from image
        person_array = image[person_top:person_bottom, person_left:person_right]
        person_pil_img = Image.fromarray(person_array)

        save_path = './people/person' + str(index) + '.jpg'

        # Temporarily save image
        person_pil_img.save(save_path)
        # Create Clarifai image object from image
        person = ClImage(file_obj=open(save_path, 'rb'))
        # Store image object
        people.append(person)

    return people

# Calculates the probability that each person's hand is raised
def detect_raised_hands(people):

    jsonResponse = hand_raised_detection.predict(people)

    prob_of_raised_hand = []
    for output in jsonResponse['outputs']:
         prob_of_raised_hand.append(output['data']['concepts'][0]['value'])

    return prob_of_raised_hand


def main():
    vidcap = cv2.VideoCapture('input.mp4')

    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        # save frame as JPEG file
        cv2.imwrite('video_frames/frame%d.jpg' % count, image)
        count += 1

    UPDATE_RATE = 5
    images = []
    faces = []
    prob_of_raised_hand = []
    for i in range(count):
        image_path = 'video_frames/frame%d.jpg' % i

        if(i % UPDATE_RATE == 0):
            #### MAKE ASYNC #####
            # Get face locations
            image_file = face_recognition.load_image_file(image_path)
            faces = face_recognition.face_locations(image_file)
            print("Frame " + str(i) +": "+ str(faces))

            people = extract_people(faces,image_file)
            prob_of_raised_hand = detect_raised_hands(people)

        img = cv2.imread(image_path)

        for index,face in enumerate(faces):
            top, right, bottom, left = face

            if prob_of_raised_hand[index] > 0.6:
                box_color = (124,252,0)
            else:
                box_color = (255,0,0)

            line_thickness = 3

            cv2.rectangle(img,(left,top),(right,bottom),box_color,line_thickness)

        images.append(img)
        #cv2.imwrite(image_path,img)

    image_height,image_width,layers = images[0].shape
    video = cv2.VideoWriter('result.MP4',-1,30,(image_width,image_height))

    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__": main()
