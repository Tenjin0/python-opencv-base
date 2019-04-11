import os
import sys
import cv2
from glob import glob
from os import path

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)

from helpers.detect_cars import train_cars

currentDirectory = path.dirname(os.path.abspath(__file__))
trainingDirectory = path.join(currentDirectory, "training")
cars_lightDir = path.join(trainingDirectory, "cars_light")
training_images = path.join(cars_lightDir + "/*.pgm")
print(training_images)
print(glob(path.join(cars_lightDir)))
# img = cv2.imread(path.join(currentDirectory, "..", "images", "1_original.jpg"))

# fn = ""
# train_cars()