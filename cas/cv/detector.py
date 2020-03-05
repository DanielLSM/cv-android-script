#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2

import time
# from absl import app, flags, logging
# from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny, Input, Darknet, YoloOutput, YoloConv
# from yolov3_tf2.models import Model, Lambda, yolo_boxes, YoloConvTiny, DarknetTiny
# from yolov3_tf2.models import Model, Lambda, yolo_boxes, YoloConvTiny, DarknetTiny
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

###############################################################################
###############################################################################
# Need to re-write yolo_nms because it had a unhealthy dependancy on the FLAGS module
# def yolo_nms(outputs,
#              anchors,
#              masks,
#              classes,
#              yolo_max_boxes=100,
#              yolo_iou_threshold=0.5,
#              yolo_score_threshold=0.5):
#     # boxes, conf, type
#     b, c, t = [], [], []

#     for o in outputs:
#         b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
#         c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
#         t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

#     bbox = tf.concat(b, axis=1)
#     confidence = tf.concat(c, axis=1)
#     class_probs = tf.concat(t, axis=1)

#     scores = confidence * class_probs
#     boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#         boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
#         scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
#         max_output_size_per_class=yolo_max_boxes,
#         max_total_size=yolo_max_boxes,
#         iou_threshold=yolo_iou_threshold,
#         score_threshold=yolo_score_threshold)

#     return boxes, scores, classes, valid_detections

###############################################################################
###############################################################################

###############################################################################
###############################################################################
#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2
# def YoloV3(size=None,
#            channels=3,
#            anchors=yolo_anchors,
#            masks=yolo_anchor_masks,
#            classes=80,
#            training=False):
#     x = inputs = Input([size, size, channels], name='input')
#     x_36, x_61, x = Darknet(name='yolo_darknet')(x)
#     x = YoloConv(512, name='yolo_conv_0')(x)
#     output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)
#     x = YoloConv(256, name='yolo_conv_1')((x, x_61))
#     output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)
#     x = YoloConv(128, name='yolo_conv_2')((x, x_36))
#     output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

#     if training:
#         return Model(inputs, (output_0, output_1, output_2), name='yolov3')

#     boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
#                      name='yolo_boxes_0')(output_0)
#     boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
#                      name='yolo_boxes_1')(output_1)
#     boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
#                      name='yolo_boxes_2')(output_2)

#     outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
#                      name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

#     return Model(inputs, outputs, name='yolov3')

# def YoloV3Tiny(size=None,
#                channels=3,
#                anchors=yolo_tiny_anchors,
#                masks=yolo_tiny_anchor_masks,
#                classes=80,
#                training=False):
#     x = inputs = Input([size, size, channels], name='input')
#     x_8, x = DarknetTiny(name='yolo_darknet')(x)
#     x = YoloConvTiny(256, name='yolo_conv_0')(x)
#     output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)
#     x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
#     output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

#     if training:
#         return Model(inputs, (output_0, output_1), name='yolov3')

#     boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
#                      name='yolo_boxes_0')(output_0)
#     boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
#                      name='yolo_boxes_1')(output_1)
#     outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
#                      name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
#     return Model(inputs, outputs, name='yolov3_tiny')

###############################################################################
###############################################################################
#main implementation of yolov3: https://github.com/zzh8829/yolov3-tf2


class Detector:
    def __init__(self,
                 classes='../../../yolov3_tf2/data/coco.names',
                 weights='../../../yolov3_tf2/checkpoints/yolov3.tf',
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
