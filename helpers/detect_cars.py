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

kaze = cv2.KAZE_create()


def extract_kaze():
    pass


def train_car():
    pass


def train_cars(pos_glob, neg_glob):

    flann_params = dict(algorithm=1, trees=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})
    extract_bow = cv2.BOWImgDescriptorExtractor(kaze, flann)

    for i in range(SAMPLES):
        if i <= len(pos_glob):
            fn = pos_glob[i]
            im = cv2.imread(fn, 0)
            descriptor = kaze.detectAndCompute(im, None)[1]
            # bow_kmeans_trainer.add( )
