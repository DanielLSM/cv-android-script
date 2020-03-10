import cv2
import vlc
import requests
import numpy as np
import tensorflow as tf
from cas.cv.detector import Detector

if __name__ == "__main__":

    classes_path = '../../yolov3-tf2/data/coco.names'
    weights_path = '../../yolov3-tf2/checkpoints/yolov3.tf'
    detector = Detector(classes=classes_path, weights=weights_path)
    # detector = Detector()

    # import ipdb
    # ipdb.set_trace()
    http_android = "http://130.229.165.230:8080/"
    # p = vlc.MediaPlayer(http_android + "audio.wav")
    # p.play()
    while True:
        url = http_android + "shot.jpg"
        img_shot = requests.get(url)
        img_array = np.array(bytearray(img_shot.content), dtype=np.uint8)
        img_raw = tf.image.decode_image(img_shot.content, channels=3)

        #original        img = cv2.imdecode(img_array, -1)
        boxes, scores, classes, nums = detector.get_inference(img_raw)
        image = detector.draw_output_image(img_raw, boxes, scores, classes, nums)
        # detector.save_output_image(image, boxes, scores, classes, nums, output_path)
        # detector.print_classfication_scores(boxes, scores, classes, nums)

        #original  cv2.imshow("AndroidCam", img)
        cv2.imshow("AndroidCam", image)

        if cv2.waitKey(1) == 27:
            break
