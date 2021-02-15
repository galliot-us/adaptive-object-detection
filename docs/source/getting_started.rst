Getting Started
===============
You can run Object  Detection module on various platforms

X86
^^^

You should have `Docker <https://docs.docker.com/get-docker/>`_ on your system. ::

    # 1) Build Docker image
    docker build -f x86.Dockerfile -t "neuralet/object-detection:latest-x86_64_cpu" .

    # 2) Run Docker container:
    docker run -it -v "$PWD":/repo neuralet/object-detection:latest-x86_64_cpu

X86 nodes with GPU
^^^^^^^^^^^^^^^^^^

You should have `Docker <https://docs.docker.com/get-docker/>`_ and `Nvidia Docker Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ on your system. ::

    # 1) Build Docker image
    docker build -f x86-gpu.Dockerfile -t "neuralet/object-detection:latest-x86_64_gpu" .

    # 2) Run Docker container:
    Notice: you must have Docker >= 19.03 to run the container with `--gpus` flag.
    docker run -it --gpus all -v "$PWD":/repo neuralet/object-detection:latest-x86_64_gpu


Nvidia Jetson Devices
^^^^^^^^^^^^^^^^^^^^^

You need to have JetPack 4.3 installed on your Jetson device. ::

    # 1) Build Docker image
    docker build -f jetson-nano.Dockerfile -t "neuralet/object-detection:latest-jetson_nano" .

    # 2) Run Docker container:
    Notice: you must have Docker >= 19.03 to run the container with `--gpus` flag.
    docker run -it --runtime nvidia --privileged -v "$PWD":/repo neuralet/object-detection:latest-jetson_nano

Coral Dev Board
^^^^^^^^^^^^^^^

::

    # 1) Build Docker image
    docker build -f coral-dev-board.Dockerfile -t "neuralet/object-detection:latest-coral-dev-board" .

    # 2) Run Docker container:
    docker run -it --privileged -v "$PWD":/repo neuralet/object-detection:latest-coral-dev-board

AMD64 node with a connected Coral USB Accelerator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # 1) Build Docker image
    docker build -f amd64-usbtpu.Dockerfile -t "neuralet/object-detection:latest-amd64" .

    # 2) Run Docker container:
    docker run -it --privileged -v "$PWD":/repo neuralet/object-detection:latest-amd64
