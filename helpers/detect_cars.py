import cv2
import numpy as np
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))
imagepath = normpath(join(currentDirectory, "..", "images"))

SAMPLES = 400

pos, neg = "pos-", "neg-"

bow_kmeans_trainer = cv2.BOWKMeansTrainer(1000)
