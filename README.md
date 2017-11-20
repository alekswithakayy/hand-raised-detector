# hand-raised-detector
A system for detecting raised hands in a lecture room.

The program locates individuals through face detection using the
ageitgey/face_recognition library. Based on the location and dimensions of the faces,
the individuals' upper body is extracted from the image and sent to a Clarifai
classification model.

The model was trained to differentiate between individuals with their hands raised
and those without their hands raised. Using the probability value returned by
Clarifai, a blue (negative) or green (affirmative) bounding box is draw around
the individuals' faces. This process is performed on each frame of a video.
