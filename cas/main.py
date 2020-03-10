import cv2
import vlc
import requests
import numpy as np
import tensorflow as tf
from cas.cv.detector import Detector
from cas.connector.pmsg import FrameCapturer

# import ipdb
# ipdb.set_trace()
if __name__ == "__main__":

    classes_path = '../../yolov3-tf2/data/coco.names'
    weights_path = '../../yolov3-tf2/checkpoints/yolov3.tf'
    url = "http://130.229.170.85:8080/video"
    detector = Detector(classes=classes_path, weights=weights_path)
    capturer = FrameCapturer(url)
    # a = FrameCapturer(0)

    while True:
        # url = http_android + "shot.jpg"
        # img_shot = requests.get(url)
        # img_array = np.array(bytearray(img_shot.content), dtype=np.uint8)

        frame, subframe = capturer.get_frame()
        # cv2.imshow('sub_frame', subframe)
        # cv2.imshow('frame', frame)
        # img_array = np.array(bytearray(subframe.content), dtype=np.uint8)
        # subframe_raw = tf.image.decode_image(subframe, channels=3)
        # subframe_raw = subframe
        #original        img = cv2.imdecode(img_array, -1)
        boxes, scores, classes, nums = detector.get_inference(subframe)
        subframe = detector.draw_output_image(subframe, boxes, scores, classes, nums, color=1)
        # detector.save_output_image(image, boxes, scores, classes, nums, output_path)
        # detector.print_classfication_scores(boxes, scores, classes, nums)

        #original  cv2.imshow("AndroidCam", img)
        # cv2.imshow("AndroidCam", image)
        cv2.imshow('sub_frame', subframe)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
