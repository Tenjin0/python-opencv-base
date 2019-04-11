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


def train_cars(fn):

    im = cv2.imread(fn, 0)
    descriptor = kaze.detectAndCompute(im, None)[1]
    descriptor2 = kaze.compute(im, kaze.detect(im))[1]
    print(descriptor)
    print(descriptor2)
    bow_kmeans_trainer.add(descriptor)
