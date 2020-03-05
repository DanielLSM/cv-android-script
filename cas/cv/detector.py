#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2

import time
# from absl import app, flags, logging
# from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

# flags.DEFINE_string('classes', '../yolov3_tf2/data/coco.names', 'path to classes file')
# flags.DEFINE_string('weights', '../yolov3_tf2/checkpoints/yolov3.tf', 'path to weights file')
# flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_string('video', '../yolov3_tf2/data/video.mp4',
#                     'path to video file or number for webcam)')
# flags.DEFINE_string('output', None, 'path to output video')
# flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
# flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


class Detector:
    def __init__(self,
                 classes='../yolov3_tf2/data/coco.names',
                 weights='../yolov3_tf2/checkpoints/yolov3.tf',
                 tiny=False,
                 size=416,
                 video='../yolov3_tf2/data/video.mp4',
                 output=None,
                 output_format='XVID',
                 num_classes=80):

        self.classes = classes
        self.weights = weights
        self.tiny = tiny
        self.size = size
        self.video = video
        self.ouput = output
        self.output_format = output_format
        self.num_classes = num_classes

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

        times = []

        # try:
        #     vid = cv2.VideoCapture(int(self.video))
        # except:
        #     vid = cv2.VideoCapture(self.video)

        out = None

        if self.output:
            # by default VideoCapture returns float instead of int
            self.width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(vid.get(cv2.CAP_PROP_FPS))
            self.codec = cv2.VideoWriter_fourcc(*self.output_format)
            self.out = cv2.VideoWriter(self.output, codec, fps, (width, height))


if __name__ == '__main__':
    detector = Detector()
    import ipdb
    ipdb.set_trace()
