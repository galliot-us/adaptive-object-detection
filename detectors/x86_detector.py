from .base_detector import BaseDetector
from protos.pedestrians_pb2 import Bbox, Person, Frame
from utils.fps_calculator import convert_infr_time_to_fps
import tensorflow as tf
import numpy as np
import cv2 as cv
import time
import logging
import os
import wget
import tarfile
import pathlib


class X86Detector(BaseDetector):
    
    def load_model(self, model_path=None):

        if not model_path:
            logging.info("you didn't specify the model file so the COCO pretrained model will be used")
            model_name = "ssd_mobilenet_v2_coco_2018_03_29"
            base_url = "http://download.tensorflow.org/models/object_detection/"
            model_file = model_name + ".tar.gz"
            base_dir = "detectors/data/"
            model_path = os.path.join(base_dir, model_name)
            if not os.path.isdir(model_path):
                logging.info('model does not exist under: {}, downloading from {}'.format(str(model_path), base_url + model_file))
                os.makedirs(base_dir, exist_ok=True)
                wget.download(base_url + model_file, base_dir)
                with tarfile.open(base_dir + model_file, "r") as tar:
                    tar.extractall(path=base_dir)

        model_dir = pathlib.Path(model_path) / "saved_model"
        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']
        self.model = model

    def preprocess(self, raw_image):
        resized_image = cv.resize(raw_image, (self.width, self.height))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        return rgb_resized_image


    def inference(self, preprocessed_image):
        """
        inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding detection output which is used for creating result
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)
        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        if not self.model:
            raise RuntimeError("first load the model with 'load_model()' method then call inferece()")
        input_image = np.expand_dims(preprocessed_image, axis=0)
        input_tensor = tf.convert_to_tensor(input_image)
        t_begin = time.perf_counter()
        output_dict = self.model(input_tensor)
        inference_time = time.perf_counter() - t_begin  # Seconds

        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)

        boxes = output_dict['detection_boxes']
        labels = output_dict['detection_classes']
        scores = output_dict['detection_scores']

        class_id = 1 
        frame = Frame(width=self.width, height=self.height, fps=self.fps) 
        for i in range(boxes.shape[1]):  # number of boxes
            if labels[0, i] == class_id and scores[0, i] > self.thresh:
                left = boxes[0, i, 1]
                top = boxes[0, i, 0]
                right = boxes[0, i, 3]
                bottom = boxes[0, i, 2]
                score = scores[0, i]
                frame.people.append(Person(
                                            id=str(class_id) + '-' + str(i),
                                            bbox=Bbox(left=left, top=top, right=right, bottom=bottom, score=score)
                                            )
                                    )

        return frame
