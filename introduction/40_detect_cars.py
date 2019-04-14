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
trainingDirectory = os.path.abspath(path.join(currentDirectory, "..", "training"))
cars_lightDir = path.join(trainingDirectory, "cars")

pos, neg = "pos-", "neg-"

cars_training_images = path.join(cars_lightDir + "/*.pgm")
glob = glob(path.join(cars_training_images))


def positive_glob(fn):

    filename = os.path.basename(fn)
    return filename.startswith(pos)


def negative_glob(fn):

    filename = os.path.basename(fn)
    return filename.startswith(neg)


pos_glog_filter = filter(positive_glob, glob)
pos_glog = list(pos_glog_filter)

neg_glog_filter = filter(negative_glob, glob)
neg_glog = list(neg_glog_filter)

train_cars(pos_glog, neg_glog)
