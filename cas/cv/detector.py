#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2

import time
import cv2
import numpy as np
import tensorflow as tf
import logging
from os import path
###############################################################################
###############################################################################
#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


class Detector:
    def __init__(
        self,
        classes='../../../yolov3_tf2/data/coco.names',
        weights='../../../yolov3_tf2/checkpoints',
        #  weights='../../../yolov3_tf2/checkpoints/yolov3.tf',
        tiny=False,
        size=416,
        video='../../../yolov3_tf2/data/video.mp4',
        output_format='XVID',
        tf_record=None,
        num_classes=80):

        assert path.exists(weights)
        assert path.exists(classes)

        self.classes = classes
        self.weights = weights
        self.tiny = tiny
        self.size = size
        self.video = video
        self.output_format = output_format
        self.tf_record = tf_record
        self.num_classes = num_classes
        self.default_output = './output.jpg'

        self.logging = logging.getLogger()
        self.logging.setLevel(logging.DEBUG)

        self.physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(self.physical_devices) > 0:
            tf.config.experimental.set_memory_growth(self.physical_devices[0], True)

        if self.tiny:
            import ipdb
            ipdb.set_trace()
            self.yolo = YoloV3Tiny(classes=self.num_classes)
        else:
            self.yolo = YoloV3(classes=self.num_classes)

        self.yolo.load_weights(self.weights)
        self.logging.info('weights loaded')

        self.class_names = [c.strip() for c in open(self.classes).readlines()]
        self.logging.info('classes loaded')

    def get_img(self, image_path=None):
        if self.tfrecord or image_path:
            #gets random images
            dataset = load_tfrecord_dataset(self.tfrecord, self.classes, self.size)
            dataset = dataset.shuffle(512)
            img_raw, _label = next(iter(dataset.take(1)))
        else:
            #gets inputed image with a path
            img_raw = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)

        return img_raw

    def get_inference(self, img_raw):
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, self.size)
        boxes, scores, classes, nums = self.yolo(img)
        return boxes, scores, classes, nums

    def print_classfication_scores(self, boxes, scores, classes, nums):
        self.logging.info('detections:')
        for i in range(nums[0]):
            self.logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                    np.array(scores[0][i]), np.array(boxes[0][i])))

    def draw_output_image(self, img_raw):
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        return img

    def save_output_image(self, img, boxes, scores, classes, nums, output_path=None):
        if not output_path:
            output_path = self.default_output
        cv2.imwrite(output_path, img)
        self.logging.info('output saved to: {}'.format(output_path))


if __name__ == '__main__':

    image_path = '../../../yolov3_tf2/data/girl.png'
    output_path = './output.jpg'
    detector = Detector()
    image_raw = detector.get_img(image_path)
    boxes, scores, classes, nums = detector.get_inference(image_raw)
    image = detector.draw_output_image(image_raw)
    detector.save_output_image(image, boxes, scores, classes, nums, output_path)
    detector.print_classfication_scores(boxes, scores, classes, nums)
