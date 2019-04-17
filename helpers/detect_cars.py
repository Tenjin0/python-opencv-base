import cv2
import numpy as np
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))
imagepath = normpath(join(currentDirectory, "..", "images"))

SAMPLES = 100

pos, neg = "pos-", "neg-"


kaze = cv2.KAZE_create()
detect = cv2.KAZE_create()


def extract_kaze():
    pass


def train_car():
    pass


def get_flann_matcher():
    flann_params = dict(algorithm=1, trees=5)
    return cv2.FlannBasedMatcher(flann_params, {})


def train_cars(pos_glob, neg_glob):

    flann = get_flann_matcher()

    bow_kmeans_trainer = cv2.BOWKMeansTrainer(1000)
    extract_bow = cv2.BOWImgDescriptorExtractor(kaze, flann)

    traindata, trainlabels = [], []

    for i in range(SAMPLES):

        if i <= len(pos_glob):
            fn = pos_glob[i]
            print(i, fn)
            impos = cv2.imread(fn, 0)
            descriptor = kaze.detectAndCompute(impos, None)[1]
            bow_kmeans_trainer.add(descriptor)
            bow_descriptor = extract_bow.compute(impos, detect.detect(impos, None))
            traindata.extend(bow_descriptor)
            trainlabels.append(1)

        # if i <= len(neg_glob):
        #     fn = neg_glob[i]
        #     imneg = cv2.imread(fn, 0)
        #     bow_descriptor = extract_bow.compute(imneg, kaze.detect(imneg))
        #     traindata.extend(bow_descriptor)
        #     trainlabels.append(-1)

            # traindata.extend(bow_features(cv2.imread(
            #     path(pos, i), 0), extract_bow, detect))
        # bow_kmeans_trainer.add( )

    # voc = bow_kmeans_trainer.cluster()
    # extract_bow.setVocabulary(voc)

    # svm = cv2.ml.SVM_create()
    # svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setGamma(1)
    # svm.setC(35)
    # svm.setKernel(cv2.ml.SVM_RBF)
    # svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

    # return svm, extract_bow

