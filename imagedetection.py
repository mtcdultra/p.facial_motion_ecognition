import cv2
from deepface import DeepFace
#import numpy as np

imgpath = 'ft2.jpg'
#imgpath = 'mm.mov'
image = cv2.imread(imgpath)
#image = cv2.VideoCapture(imgpath)



analyze = DeepFace.analyze(image,actions=['emotion'])  #same thing is happing here as the previous example, we are using the analyze class from deepface and using ‘frame’ as input
analyze['dominant_emotion']  #here we will only go print out the dominant emotion also explained in the previous example


print(analyze)