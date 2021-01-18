from .base_detector import BaseDetector
from protos.pedestrians_pb2 import Bbox, Person, Frame
from utils.fps_calculator import convert_infr_time_to_fps
import logging
import time
import os
import numpy as np
import wget
import cv2 as cv
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter


class EdgeTpuDetector(BaseDetector):

    def load_model(self, model_path=None):
        if not model_path:
            logging.info("you didn't specify the model file so the COCO pretrained model will be used")
            base_url = 'https://media.githubusercontent.com/media/neuralet/neuralet-models/master/edge-tpu/mobilenet_ssd_v2/'
            model_file = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
            base_dir = "detectors/data"
            model_path = os.path.join(base_dir, model_file) 
            if not os.path.isfile(model_path):
                logging.info('model does not exist under: {}, downloading from {}'.format(str(model_path), base_url + model_file))
                os.makedirs(base_dir, exist_ok=True)
                wget.download(base_url + model_file, model_path)
        self.model = Interpreter(model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")])
        self.model.allocate_tensors()
        # Get the model input and output tensor details
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def preprocess(self, raw_image):
        resized_image = cv.resize(raw_image, (self.width, self.height))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        return rgb_resized_image

    def inference(self, preprocessed_image):
        if not self.model:
            raise RuntimeError("first load the model with 'load_model()' method then call inferece()")
        input_image = np.expand_dims(preprocessed_image, axis=0)
        # Fill input tensor with input_image
        self.model.set_tensor(self.input_details[0]["index"], input_image)
        t_begin = time.perf_counter()
        self.model.invoke()
        inference_time = time.perf_counter() - t_begin  # Second
        self.fps = convert_infr_time_to_fps(inference_time)
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        boxes = self.model.get_tensor(self.output_details[0]['index'])
        labels = self.model.get_tensor(self.output_details[1]['index'])
        scores = self.model.get_tensor(self.output_details[2]['index'])
        class_id = 0
        frame = Frame(width=self.width, height=self.height, fps=self.fps)
        for i in range(boxes.shape[1]):  # number of boxes
            if labels[0, i] == class_id and scores[0, i] > self.thresh:
                left = boxes[0, i, 1]
                top = boxes[0, i, 0]
                right = boxes[0, i, 3]
                bottom = boxes[0, i, 2]
                score = scores[0, i]
                frame.people.append(
                                    Person(id=str(class_id) + '-' + str(i),
                                            bbox=Bbox(left=left, top=top, right=right, bottom=bottom, score=score)
                                            )
                                    ) 


        return frame
