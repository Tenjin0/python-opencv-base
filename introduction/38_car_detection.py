import cv2

import numpy as np
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))


def path(cls, i):
    return "%s/%s%d.pgm" % (datapath, cls, i)


def extract_kaze(fn):

    im = cv2.imread(fn, 0)
    return extract.detectAndCompute(im, None)[1]

# img = cv2.imread(os.path.join(currentDirectory, "..", "images", "1a_original.jpg"))

detect = cv2.KAZE_create()
extract = cv2.KAZE_create()

flann_params = dict(algorithm=1, trees=5)
flann = cv2.FlannBasedMatcher(flann_params, {})

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)

extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

pos, neg = "pos-", "neg-"

for i in range(8):

    bow_kmeans_trainer.add(extract_kaze(path(pos, i)))
    bow_kmeans_trainer.add(extract_kaze(path(neg, i)))

voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)


def bow_features(fn):
    im = cv2.imread(fn, 0)
    return extract_bow.compute(img, detect.detectAndCompute(img, None))

traindata, trainlabels = [], []