# Neuralet Object Detection
## Introduction
This is the neuralet's object detection repository. With this module you can run object detection models on various edge devices and create an adaptive learning session to customize object detection models to your specific environment.
for more information please visit our [website](https://neuralet.com/) or reach out to hello@neuralet.com.

## Getting Started:
### X86 nodes with GPU:
You should have [Docker](https://docs.docker.com/get-docker/) and [Nvidia Docker Toolkit](https://github.com/NVIDIA/nvidia-docker) on your system.
```
# 1) Build Docker image
docker build -f x86.Dockerfile -t "neuralet/object-detection:latest-x86_64_gpu" .

# 2) Run Docker container:
Notice: you must have Docker >= 19.03 to run the container with `--gpus` flag.
docker run -it --gpus all -v "$PWD":/repo neuralet/object-detection:latest-x86_64_gpu
```


### Nvidia Jetson Devices

You need to have JetPack 4.3 installed on your Jetson device.

```
# 1) Build Docker image
docker build -f jetson-nano.Dockerfile -t "neuralet/object-detection:latest-jetson_nano" .

# 2) Run Docker container:
Notice: you must have Docker >= 19.03 to run the container with `--gpus` flag.
docker run -it --runtime nvidia --privileged -v "$PWD":/repo neuralet/object-detection:latest-jetson_nano
```

### Coral Dev Board

```
# 1) Build Docker image
docker build -f coral-dev-board.Dockerfile -t "neuralet/object-detection:latest-coral-dev-board" .

# 2) Run Docker container:
docker run -it --privileged -v "$PWD":/repo neuralet/object-detection:latest-coral-dev-board
```

### AMD64 node with a connected Coral USB Accelerator

```
# 1) Build Docker image
docker build -f amd64-usbtpu.Dockerfile -t "neuralet/object-detection:latest-amd64" .

# 2) Run Docker container:
docker run -it --privileged -v "$PWD":/repo neuralet/object-detection:latest-amd64
```

## Export Models to Edge Devices
### Compile tflite Models to Edge TPU Models
In amd64 with connected USB Accelerator docker container run:
`python3 exporters/edgetpu_exporter.py --tflite_file TFLITE_FILE --out_dir OUT_DIR`

### Export TensorFlow protobuf models to TRT engines
In Jetson with connected USB Accelerator docker container run:
`python3 exporters/trt_exporter.py --pb_file PB_FILE --out_dir OUT_DIR [ --num_classes  NUM_CLASSES]`

## RUN Inference
On any of the docker containers you can run sample inference to get an output video:
```
python3 inference.py --device DEVICE --input_video INPUT_VIDEO --out_dir OUT_DIR \
                    [--model_path MODEL_PATH] [--threshold THRESHOLD]  [--input_width INPUT_WIDTH]\
                    [--input_height INPUT_HEIGHT] [--out_width OUT_WIDTH] [--out_height OUT_HEIGHT]
```
Note that if you do not provide the model file, the default COCO model will be used.

## Adaptive Learning

Adaptive learning is the process of customization of object detection models with user provided data and environments. For more information please visit our [blog post](https://neuralet.com/article/adaptive-learning/).
