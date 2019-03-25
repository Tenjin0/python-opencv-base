import cv2

import numpy as np
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))


def path(cls, i):
    return "%s/%s%d.pgm" % (datapath, cls, i)


def extract_orb(fn):

    im = cv2.imread(fn, 0)
    print(extract.detectAndCompute(im, None))
    return extract.detectAndCompute(im)


# img = cv2.imread(os.path.join(currentDirectory, "..", "images", "1a_original.jpg"))

detect = cv2.ORB_create()
extract = cv2.ORB_create()

flann_params = dict(algorithm=1, trees=5)
flann = cv2.FlannBasedMatcher(flann_params, {})

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)

extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

pos, neg = "pos-", "neg-"

for i in range(8):
    bow_kmeans_trainer.add(extract_orb(path(pos, i)))
    # bow_kmeans_trainer.add(extract_orb(path(neg, i)))

# voc = bow_kmeans_trainer.cluster()
# extract_bow.setVocabulary(voc)


# def bow_features(fn):
#     im = cv2.imread(fn, 0)
#     return extract_bow.compute(img, detect.detectAndCompute(img, None))

# traindata, trainlabels = [], []