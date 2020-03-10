#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import logging
from os import path

# disables TF print log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
###############################################################################
###############################################################################
#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


class Detector:
    def __init__(self,
                 classes='../../../yolov3-tf2/data/coco.names',
                 weights='../../../yolov3-tf2/checkpoints/yolov3.tf',
                 tiny=False,
                 size=416,
                 video='../../../yolov3-tf2/data/video.mp4',
                 output_format='XVID',
                 tfrecord=None,
                 num_classes=80):

        self.classes = classes
        self.weights = weights
        self.tiny = tiny
        self.size = size
        self.video = video
        self.output_format = output_format
        self.tfrecord = tfrecord
        self.num_classes = num_classes
        self.default_output = './output.jpg'

        self.logging = logging
        self.logging.basicConfig(level=logging.NOTSET)

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
        if self.tfrecord or not image_path:
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
            self.logging.info('\t{}, {}, {}'.format(self.class_names[int(classes[0][i])],
                                                    np.array(scores[0][i]), np.array(boxes[0][i])))

    def draw_output_image(self, img_raw, boxes, scores, classes, nums, color=0):
        img_raw = img_raw if isinstance(img_raw, np.ndarray) else img_raw.numpy()
        img = img_raw if color else cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
        return img

    def save_output_image(self, img, boxes, scores, classes, nums, output_path=None):
        if not output_path:
            output_path = self.default_output
        cv2.imwrite(output_path, img)
        self.logging.info('output saved to: {}'.format(output_path))


if __name__ == '__main__':

    image_path = '../../../yolov3-tf2/data/girl.png'
    output_path = './output.jpg'
    detector = Detector()
    image_raw = detector.get_img(image_path)
    boxes, scores, classes, nums = detector.get_inference(image_raw)
    image = detector.draw_output_image(image_raw, boxes, scores, classes, nums)
    detector.save_output_image(image, boxes, scores, classes, nums, output_path)
    detector.print_classfication_scores(boxes, scores, classes, nums)
