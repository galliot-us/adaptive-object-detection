from .base_detector import BaseDetector
from protos.objects_bboxes_pb2 import Bbox, Instance, Frame
import cv2 as cv
import ctypes
import logging
import os
import wget
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import time
from pathlib import Path
from utils.fps_calculator import convert_infr_time_to_fps


class JetsonDetector(BaseDetector):
    """
    Perform object detection with the given prebuilt tensorrt engine.
    """

    def _load_plugins(self):
        """ Required as Flattenconcat is not natively supported in TensorRT. """
        ctypes.CDLL("/opt/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self, model_path, classes):
        parent_dir = str(Path(__file__).parent.absolute())
        base_dir = parent_dir + "/data/"
        detector_class_count = len(classes)
        if not model_path:
            logging.info("you didn't specify the model file so the COCO pretrained model will be used")
            base_url =  "https://github.com/Tony607/jetson_nano_trt_tf_ssd/raw/master/packages/jetpack4.3/"
            model_file = "frozen_inference_graph.bin"
            model_path = os.path.join(base_dir, model_file)
            if not os.path.isfile(model_path):
                logging.info('model does not exist under: {}, downloading from {}'.format(str(model_path), base_url + model_file))
                os.makedirs(base_dir, exist_ok=True)
                os.system("bash "+ parent_dir + "/../generate_tensorrt.bash")
        import pathlib
        if ( pathlib.Path(model_path).suffix == ".pb" ):
            logging.info('model is a Tensorflow protobuf... Converting...')
            os.makedirs(base_dir, exist_ok=True)
            os.system("bash "+ parent_dir + "/../generate_tensorrt.bash " + str(model_path) + " " + str(detector_class_count))
            model_file = "frozen_inference_graph.bin"
            model_path = os.path.join(base_dir, model_file)

        """ Load engine file as a trt Runtime. """
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """
        Create some space to store intermediate activation values. 
        Since the engine holds the network definition and trained parameters, additional space is necessary.
        """
        for binding in self.model:
            size = trt.volume(self.model.get_binding_shape(binding)) * \
                   self.model.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.model.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                
            del host_mem
            del cuda_mem
 
        return

    def __init__(self, *args, **kwargs):
        super(JetsonDetector, self).__init__(*args, **kwargs)
        self.cuda_inputs = None
        self.cuda_outputs = None
        self.stream = None
        self.cuda_outputs = None
        self.cuda_inputs = None
        self.cuda_context = None
        self.engine_context = None
        self.bindings = None
        self.host_inputs = None
        self.host_outputs = None


    def load_model(self, model_path=None, label_map=None):
        """ 
        Initialize TensorRT plugins, engine and context with given tensorrt engine file, if no file provided the default
        COCO model will be downloaded under 'detectors/data'.
        Args:
            model_path: Path to the tensorrt engine.
        """
 
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.classes = list(label_map.keys())
        self._init_cuda_stuff(model_path)
        self.label_map = label_map


    def _init_cuda_stuff(self, model_path):
        cuda.init()
        self.device = cuda.Device(0)  # enter your Gpu id here
        self.cuda_context = self.device.make_context()
        self.model = self._load_engine(model_path, self.classes) 
        self._allocate_buffers()
        self.engine_context = self.model.create_execution_context()
        self.stream = cuda.Stream()  # create a CUDA stream to run inference

    def __del__(self):
        """ Free CUDA memories. """
        for mem  in self.cuda_inputs:
            mem.free()
        for mem in self.cuda_outputs:
            mem.free

        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs
        self.cuda_context.pop()
        del self.cuda_context
        del self.engine_context
        del self.model
        del self.bindings
        del self.host_inputs
        del self.host_outputs

    @staticmethod
    def _preprocess_trt(img):
        """ Preprocess an image before TRT SSD inferencing. """
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = (2.0 / 255.0) * img - 1.0
        return img
    def preprocess(self, raw_image):
        """
        preprocess function prepares the raw input for inference.
        Args:
            raw_image: A BGR numpy array with shape (img_height, img_width, channels)
        Returns:
            rgb_resized_image: A numpy array which contains preprocessed verison of input
        """

        resized_image = cv.resize(raw_image, (self.width, self.height))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        rgb_resized_image = self._preprocess_trt(rgb_resized_image)
        return rgb_resized_image 

    def _postprocess_trt(self, output):
        """ Postprocess TRT SSD output. """ 
        boxes, confs, clss = [], [], []
        for prefix in range(0, len(output), 7):
            # index = int(output[prefix+0])
            conf = float(output[prefix + 2])
            if conf < float(self.thresh):
                continue
            x1 = (output[prefix + 3])  
            y1 = (output[prefix + 4]) 
            x2 = (output[prefix + 5])
            y2 = (output[prefix + 6])
            cls = int(output[prefix + 1])
            boxes.append((y1, x1, y2, x2))
            confs.append(conf)
            clss.append(cls)
        return boxes, confs, clss

    def inference(self, preprocessed_image):
        """
        Detect objects in the input image.
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)
        Returns:
            result: A Frame protobuf massages
       """
        if not self.model:
            raise RuntimeError("first load the model with 'load_model()' method then call inferece()")
        # transfer the data to the GPU, run inference and the copy the results back
        np.copyto(self.host_inputs[0], preprocessed_image.ravel())

        # Start inference time
        t_begin = time.perf_counter()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.engine_context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        inference_time = time.perf_counter() - t_begin  # Seconds

        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)
        output = self.host_outputs[0]
        boxes, scores, classes = self._postprocess_trt(output)
        frame = Frame(width=self.width, height=self.height, fps=self.fps)
        for i in range(len(boxes)):  # number of boxes
            if classes[i] in self.classes:
                left = boxes[i][1]
                top = boxes[i][0]
                right = boxes[i][3]
                bottom = boxes[i][2]
                score = scores[i]

                frame.objects.append(
                                    Instance(id=str(int(classes[i]))+ "-" + str(i),
                                        category=self.label_map[int(classes[i])]["name"],
                                        bbox=Bbox(left=left, top=top, right=right, bottom=bottom, score=score)
                                        )
                                    ) 

        return frame
