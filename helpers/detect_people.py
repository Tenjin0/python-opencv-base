
import time
import cv2
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np


class Detect_people:
    CAMERA_RESOLUTION = (1024, 768)
    BOX_COLOR1 = (0, 0, 255)
    BOX_COLOR2 = (0, 255, 0)

    # People very small size and close together
    CALIBRATION_MODE_1 = (400, (2, 2), (8, 8), 1.01, 0.999)
    # People very small size
    CALIBRATION_MODE_2 = (400, (3, 3), (32, 32), 1.01, 0.8)
    # People small size and close together
    CALIBRATION_MODE_3 = (400, (4, 4), (32, 32), 1.015, 0.999)
    # People small size
    CALIBRATION_MODE_4 = (400, (4, 4), (32, 32), 1.015, 0.8)
    # People medium size and close together
    CALIBRATION_MODE_5 = (400, (4, 4), (32, 32), 1.02, 0.999)
    # People medium size
    CALIBRATION_MODE_6 = (400, (4, 4), (32, 32), 1.02, 0.8)
    # People large size and close together
    CALIBRATION_MODE_7 = (400, (6, 6), (32, 32), 1.03, 0.999)
    # People large size
    CALIBRATION_MODE_8 = (400, (6, 6), (32, 32), 1.03, 0.8)

    CALIBRATION_MODE_9 = (400, (8, 8), (32, 32), 1.04, 0.999)

    CALIBRATION_MODE_10 = (400, (8, 8), (32, 32), 1.04, 0.8)

    CALIBRATION_MODES = (
        CALIBRATION_MODE_1, CALIBRATION_MODE_2,
        CALIBRATION_MODE_3, CALIBRATION_MODE_4,
        CALIBRATION_MODE_5, CALIBRATION_MODE_6,
        CALIBRATION_MODE_7, CALIBRATION_MODE_8,
        CALIBRATION_MODE_9, CALIBRATION_MODE_10
    )

    def __init__(self):
        # default mode
        self.MIN_IMAGE_WIDTH, self.WIN_STRIDE, self.PADDING, self.SCALE, self.OVERLAP_THRESHOLD = self.CALIBRATION_MODE_5
        self.SHOW_IMAGES = True
        self.IMAGE_WAIT_TIME = 0  # Wait indefinitely until button pressed
        self.DRAW_RAW_RECT = False
        self.DRAW_RECT = False
        self.CLOSE_WINDOW = True

    def set_calibration(self, idx=None, tup=None):
        '''
        Set people detection calibration with EITHER an index of a preset calibration mode, or manually set all values
        with a tuple. If passed both an index and a tuple, only the index will be considered.
        :param idx: index of a calibration mode in self.CALIBRATION_MODES
        :param tup: alternative to idx, tuple with 5 values
        :return: None
        '''
        if idx and idx < len(self.CALIBRATION_MODES):
            self.MIN_IMAGE_WIDTH, self.WIN_STRIDE, self.PADDING, self.SCALE, self.OVERLAP_THRESHOLD = self.CALIBRATION_MODES[
                idx]
        elif idx:
            raise IndexError('list index out of range')
        if tup:
            assert len(tup) == 5
            self.MIN_IMAGE_WIDTH, self.WIN_STRIDE, self.PADDING, self.SCALE, self.OVERLAP_THRESHOLD = tup

    def draw_image(self, img):
        cv2.imshow("People detection", img)
        cv2.waitKey(self.IMAGE_WAIT_TIME)
        cv2.destroyAllWindows()

    def try_all_calibration_modes(self, img):
        self.CLOSE_WINDOW = False
        for i, calibration in enumerate(self.CALIBRATION_MODES):
            self.set_calibration(i)
            image, picklen, rectlen = self.find_people(img)
            print(self.CALIBRATION_MODES[i], picklen, rectlen)
            cv2.imshow("People detection", image)
            cv2.waitKey(self.IMAGE_WAIT_TIME)

    def find_best_calibration(self, img):
        pass

    def find_people(self, img):
        '''
        Detect people in image
        :param img: numpy.ndarray
        :return: count of rectangles after non-maxima suppression, corresponding to number of people detected in picture
        '''
        t = time.time()
        # HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Chooses whichever size is less
        # image = imutils.resize(img, width=min(
        #     self.MIN_IMAGE_WIDTH, img.shape[1]))
        image = img.copy()
        # detect people in the image
        (rects, wghts) = hog.detectMultiScale(
            image,
            winStride=self.WIN_STRIDE,
            padding=self.PADDING,
            scale=self.SCALE
        )

        t2 = time.time()

        # apply non-maxima suppression to the bounding boxes but use a fairly large overlap threshold,
        # to try to maintain overlapping boxes that are separate people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        if self.DRAW_RAW_RECT:
            for (xA, yA, xB, yB) in rects:
                cv2.rectangle(image, (xA, yA), (xB, yB), self.BOX_COLOR1, 2)
        pick = non_max_suppression(
            rects, probs=None, overlapThresh=self.OVERLAP_THRESHOLD)

        print("Elapsed time: {}, {} seconds".format(
            int((time.time() - t) * 100) / 100.0,
            int((time.time() - t2) * 100) / 100.0
        ))

        if self.DRAW_RECT:
            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                # Tighten the rectangle around each person by a small margin
                shrinkW, shrinkH = int(0.05 * xB), int(0.05*yB)
                cv2.rectangle(image, (xA+shrinkW, yA+shrinkH),
                              (xB-shrinkW, yB-shrinkH), self.BOX_COLOR2, 2)

        return image, len(pick), len(rects)
