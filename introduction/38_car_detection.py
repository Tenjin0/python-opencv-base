import cv2

import numpy as np
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))
imagepath = normpath(join(currentDirectory, "..", "images"))


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

bow_kmeans_trainer = cv2.BOWKMeansTrainer(1000)

extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

pos, neg = "pos-", "neg-"
SAMPLES = 40

for i in range(SAMPLES):
    try:
        bow_kmeans_trainer.add(extract_kaze(path(pos, i)))
    except:
        print("image", pos, i)
    try:
        bow_kmeans_trainer.add(extract_kaze(path(neg, i)))
    except:
        print("image", neg, i)

voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)


def bow_features(fn):
    img = cv2.imread(fn, 0)
    return extract_bow.compute(img, detect.detect(img, None))


traindata, trainlabels = [], []

for i in range(SAMPLES):
    try:
        traindata.extend(bow_features(path(pos, i)))
        trainlabels.append(1)
    except:
        print("image", pos, i)
    try:
        traindata.extend(bow_features(path(neg, i)))
        trainlabels.append(-1)
    except:
        print("image", neg, i)

svm = cv2.ml.SVM_create()

svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))


def predict(fn):
    f = bow_features(fn)
    p = svm.predict(f)
    print(fn, "\t", p[1][0][0])
    return p


car, notcar = join(imagepath, "car.jpg"), join(imagepath, "bb.jpg")

car_img = cv2.imread(car)
notcar_img = cv2.imread(notcar)

car_predict = predict(car)
not_car_predict = predict(notcar)

font = cv2.FONT_HERSHEY_SIMPLEX

if (car_predict[1][0][0] == 1.0):
    cv2.putText(car_img, 'Car Detected', (10, 30), font, 1,
                (0, 255, 0), 2, cv2.LINE_AA)

if (not_car_predict[1][0][0] == -1.0):
    cv2.putText(notcar_img, 'Car Not Detected', (10, 30),
                font, 1, (0, 0, 255), 2, cv2.LINE_AA)


cv2.imshow("BOW + SVM Failure", notcar_img)
cv2.imshow("BOW + SVM Success", car_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
