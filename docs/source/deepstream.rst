Deploy Model Using NVIDIA DeepStream
====================================
You can perform inference using NVIDIA Deepstream to boost your model run-time speed. You need a trained model which can be a retrained custom model downloaded from the Adaptive Learning module or a pretrained general one from the `Tensorflow Model Zoo <http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz>`_.

There have been provided two versions of Deepstream modules which are compatible with different OS module versions as follows:

dGPU model Platform and OS Compatibility:


+------------+--------------+------+--------------+------------------+----------------+
| Deepstream | OS           | Cuda | Cudnn        | TensorRT release | Display Driver |
+============+==============+======+==============+==================+================+
| 5.1        | Ubuntu 18.04 | 11.1 | CuDNN 8.0+   | TRT 7.2.X        | R460.32        |
+------------+--------------+------+--------------+------------------+----------------+
| 5.0        | Ubuntu 18.04 | 10.2 | cuDNN 7.6.5+ | TRT 7.0.0        | R450.51        |
+------------+--------------+------+--------------+------------------+----------------+


Jetson model Platform and OS Compatibility:


+------------+---------------------------------------+------------------+------+---------------+----------+------------------+
| Deepstream | Jetson Platforms                      | OS               | Cuda | Cudnn         | Jetpack  | TensorRT release |
+============+=======================================+==================+======+===============+==========+==================+
| 5.1        | Nano, AGX Xavier, TX2, TX1, Jetson NX | L4T Ubuntu 18.04 | 10.2 | cuDNN 8.0.0.x | 4.5.1 GA | TRT 7.1.3        |
|            |                                       | Release          |      |               |          |                  |
|            |                                       | 32.5.1           |      |               |          |                  |
+------------+---------------------------------------+------------------+------+---------------+----------+------------------+
| 5.0        | Nano, AGX Xavier, TX2, TX1, Jetson NX | L4T Ubuntu 18.04 | 10.2 | cuDNN 8.0.0.x | 4.4 GA   | TRT 7.1.3        |
|            |                                       | Release          |      |               |          |                  |
|            |                                       | 32.4.3           |      |               |          |                  |
+------------+---------------------------------------+------------------+------+---------------+----------+------------------+


You need to change directory to the appropriate deepstream version based on your platform and OS components. ::

    cd deepstream/5.0/ 
or ::

    cd deepstream/5.1/
    
Run on x86:
***********
First, you need to copy your frozen_inference_graph which can be located in the :code:`train_outputs/frozen_graph/frozen_inference_graph.pb` path to deepstream-data directory. You can skip this step if you want to perform inference on the pretrained SSD-Mobilenet-v2-coco from the model zoo. ::

    cp train_outputs/frozen_graph/frozen_inference_graph.pb deepstream-data/

Then, you need to prepare a label file at the next step. If you have retrained a model using the adaptive learning module, there is a `label_map.pbtxt` file in the train_outputs directory. If you have trained your model with all of the coco labels, you can skip this step as the label file will be downloaded automatically. 

**5.1:**
::

    docker build -f deepstream-51-x86.Dockerfile -t "neuralet/object-detection:deepstream5.1-x86" .

    docker run -it  --runtime nvidia --gpus all -e videoPath=<video_file_path> --env labelPath= <label_map.pbtxt_file_path> -v "$PWD/../../":/repo neuralet/object-detection:deepstream5.1-x86


**5.0:**
::

    docker build -f deepstream-x86.Dockerfile -t "neuralet/object-detection:deepstream5.0-x86" .

    docker run -it --gpus all --runtime nvidia -e videoPath=<video_file_path> --env labelPath=<label_map.pbtxt_file_path> -v "$PWD/../..":/repo  neuralet/object-detection:deepstream5.0-x86



Run on Jetson Devices:
**********************
First you need to generate TensorRT using :code:`exporter/trt_exporter.py` script. It will generate a :code:`frozen_inference_graph.bin` engine file and you need to copy it inside the deepstream data directory. 

Then, you need to prepare a label file at the next step. If you have retrained a model using the adaptive learning module, there is a :code:`label_map.pbtxt` file in the train_outputs directory. If you have trained your model with all of the coco labels, you can skip this step as the label file will be downloaded automatically. 

**5.1:**
::

    docker build -f deepstream-51-jetson.Dockerfile -t "neuralet/object-detection:deepstream-jetson-4-5" .
    docker run --runtime nvidia -e videoPath=<path-to-video-file> --env labelPath=<label-pbtxt-path> --privileged -it -v "$PWD/../../":/repo neuralet/object-detection:deepstream-jetson-4-5
    

**5.0**
::

    docker build -f deepstream-jetson.Dockerfile -t "neuralet/object-detection:deepstream-jetson-4-4" .
    docker run --runtime nvidia -e videoPath=<path-to-video-file> --env labelPath=<label-pbtxt-path> --privileged -it -v $PWD/../..:/repo neuralet/object-detection:deepstream-jetson-4-4


