# import the necessary packages
import datetime
from threading import Thread
import cv2


class FPS:
    def __init__(self):
        self._start = None
        self.end = None
        self.numFrames = 0

    def start(self):

        self._start = datetime.datetime.now()

        return self

    def stop(self):
        self.end = datetime.datetime.now()

    def update(self):
        self.numFrames += 1

    def elapsed(self):
        return (self.end - self._start).total_seconds()

    def fps(self):
        return self.numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):

        return self.frame

    def stop(self):
        self.stopped = True
