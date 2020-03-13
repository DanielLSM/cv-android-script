import cv2
import vlc
import requests
import numpy as np
import tensorflow as tf
from cv.detector import Detector
from connector.pmsg import FrameCapturer
from sr.speaker import Speaker

from threading import Thread, Lock


def translate_thread():
    global recognized
    global running
    running = True
    old_recognized = False
    i = 0
    while running:
        #print(str(recognized))
        #if (recognized) and recognized != old_recognized:
        if (bool(recognized) and not i % 1000) and recognized != old_recognized:
            i = 0
            old_recognized = recognized
            speaker.dialog(recognized)
            print(i)
        print(i)
        i+=1


if __name__ == "__main__":

    classes_path = '../../yolov3-tf2/data/coco.names'
    weights_path = '../../yolov3-tf2/checkpoints/yolov3.tf'
    url = "http://130.229.145.192:8080/video"
    detector = Detector(classes=classes_path, weights=weights_path)
    speaker = Speaker(dest_language='es')
    #capturer = FrameCapturer(url)
    capturer = FrameCapturer(0) #USE THIS LINE WITH JUST WEBCAM
    running = True
    recognized = False
    speaker_thread = Thread(target=translate_thread)
    speaker_thread.start()
    while True:

        frame, subframe = capturer.get_frame()
        boxes, scores, classes, nums = detector.get_inference(subframe)
        recognized = detector.get_highest_recognized(boxes, scores, classes, nums)
        print(recognized)
        subframe = detector.draw_output_image(subframe, boxes, scores, classes, nums, color=1)

        cv2.imshow('sub_frame', subframe)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            running = False
            translate_thread.join()
            break
