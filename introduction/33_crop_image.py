
# Python 2/3 compatibility
from __future__ import print_function
import sys

import numpy as np
import cv2

# built-in modules
from collections import namedtuple

# local modules
import matplotlib.pyplot as plt

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

MIN_MATCH_COUNT = 10


class RectSelector:

    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None

    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y])  # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)

    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

    @property
    def dragging(self):
        return self.drag_rect is not None


class App:

    def __init__(self, source):
        self.cap = cap = cv2.VideoCapture(source)
        self.frame = None
        self.paused = False
        cv2.namedWindow('plane')
        self.rect_sel = RectSelector('plane', self.on_rect)
        self.crop_image = None

    def on_rect(self, rect):
        print(rect)
        x0, y0, x1, y1 = rect
        self.crop_image = self.frame[y0:y1, x0:x1]
        cv2.imwrite("./data/lipton" + "/%s" % filename, frame)
        # self.tracker.add_target(self.frame, rect)

    def run(self):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            vis = self.frame.copy()

            self.rect_sel.draw(vis)
            # if(self.crop_image is not None):
            #     cv2.imshow('crop_image', self.crop_image)
            cv2.imshow('plane', vis)
            ch = cv2.waitKey(1) & 0xFF
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break


if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()
