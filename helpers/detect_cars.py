import cv2
import numpy as np
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))
imagepath = normpath(join(currentDirectory, "..", "images"))

SAMPLES = 400

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

    bow_kmeans_trainer = cv2.BOWKMeansTrainer(500)
    extract_bow = cv2.BOWImgDescriptorExtractor(kaze, flann)

    traindata, trainlabels = [], []
    exceptions = []
    for i in range(SAMPLES):

        if i <= len(pos_glob):
            fp = pos_glob[i]
            impos = cv2.imread(fp, 0)
            descriptor = kaze.detectAndCompute(impos, None)[1]
            bow_kmeans_trainer.add(descriptor)

        if i <= len(neg_glob):
            fp = neg_glob[i]
            imneg = cv2.imread(fp, 0)
            descriptor = kaze.detectAndCompute(imneg, None)[1]
            bow_kmeans_trainer.add(descriptor)

    voc = bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(voc)

    for i in range(SAMPLES):

        if i <= len(pos_glob):
            fn = pos_glob[i]
            impos = cv2.imread(fn, 0)
            bow_descriptor = extract_bow.compute(impos, kaze.detect(impos, None))
            traindata.extend(bow_descriptor)
            trainlabels.append(1)

        if i <= len(neg_glob):
            fn = neg_glob[i]
            imneg = cv2.imread(fn, 0)
            bow_descriptor = extract_bow.compute(imneg, kaze.detect(imneg))
            traindata.extend(bow_descriptor)
            trainlabels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(1)
    svm.setC(35)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

    return svm, extract_bow

