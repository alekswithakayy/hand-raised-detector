import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import face_recognition
import threading
import shutil
import os

from PIL import Image
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from collections import namedtuple
from io import StringIO
from collections import namedtuple

IMAGE_PATH = './class3.jpg'
image = face_recognition.load_image_file(IMAGE_PATH)

face_locations = face_recognition.face_locations(image)

# clarifai_images stores the cropped out photos of people
shutil.rmtree('clarifai_images')
os.makedirs('clarifai_images')
clarifai_images = []

# Stores location of each face in the image
people = []

for index,face_location in enumerate(face_locations):

    face_top, face_right, face_bottom, face_left = face_location
    face_height = face_bottom - face_top
    face_width = face_right - face_left

    # Store top left coordinate and width/height of face
    people.append((face_left,face_top,face_width,face_height))

    image_height, image_width, _ = image.shape

    person_top = max((face_top - face_height * 3), 0)
    person_right = min((face_right + face_width * 2), image_width)
    person_bottom = min((face_bottom + face_height * 2), image_height)
    person_left = max((face_left - face_width), 0)

    # You can access the actual face itself like this:
    person_array = image[person_top:person_bottom, person_left:person_right]
    person_pil_img = Image.fromarray(person_array)

    person_pil_img.save('./clarifai_images/' + str(index) + '.jpeg')

    clarifai_image = ClImage(file_obj=open('./clarifai_images/' + str(index) + '.jpeg', 'rb'))
    clarifai_images.append(clarifai_image)

# Find if hand is raised
#***********************

app = ClarifaiApp(api_key='a41852077a72486398e063fb60936b65')
hand_raised_detection = app.models.get('hand-raised-detector')
jsonResponse = hand_raised_detection.predict(clarifai_images)

prob_of_hand_raised = []
for output in jsonResponse['outputs']:
     prob_of_hand_raised.append(output['data']['concepts'][0]['value'])

# Draw bounding box
# *****************

# Create figure and axes
fig,ax = plt.subplots(1)
# Display the image
ax.imshow(image)

for index,location in enumerate(people):
    left,top,width,height = location
    box_color = ''
    if prob_of_hand_raised[index] > 0.8:
        box_color = 'g'
    else:
        box_color = 'r'

    # Create a Rectangle patch
    rect = patches.Rectangle((left,top),width,height,linewidth=2,edgecolor=box_color,facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.savefig('result.png')
