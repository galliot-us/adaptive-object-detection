Adaptive Learning
=================

Adaptive learning is the process of customization of object detection models with user-provided data and environments. For more information, please visit our `blog post <https://neuralet.com/article/adaptive-learning/>`_.

Client
^^^^^^

Neuralet adaptive learning service includes client/server side. You can start an adaptive learning task on the cloud and get the model after training on the client-side.

Run the docker container based on your device and the below commends inside the container: ::

    cd services/adaptive-learning/client

**#Step 1:**

Create an :code:`input.zip` file from the video file you want to feed to Adaptive Learning. ::

    zip -j input.zip PATH_TO_VIDEO_FILE

**#Step 2:**

Upload the zip file and get a unique id: ::

    python3 client.py upload_file --file_path FILE_PATH

**#Step 3:**

Add the previous step's unique id to the :code:`UploadUUID` field and the video file name to the :code:`VideoFile` field of the config file. You can find a more comprehensive explanation of the config file and its fields in the next section. Note: You can use the sample config file in :code:`configs/sample_config.ini`

**#Step 4:**

Initiate a new job and get your job's ID: ::

    python3 client.py train --config_path CONFIGPATH

**#Step 5:**

Get a job status (enter the job id at JASKID) ::

    python3 client.py get_status --job_id JOBID


The expected status massages are as follows:

.. csv-table:: a title
    :header: "Parameter", "Comments"
    :widths: 10, 20

    "Allocating Resource", "Allocating compute machine to your job"
    "Building", "Building an environment to start a job"
    "Training", "Running a Adaptive Learning Job"
    "Wrapping Up", "Saving data and finishing the job"
    "Finished", "The job has been finished. Note that it doesn't mean that the job has been finished successfully. it may finished with error"
    "Failed", "There was a problem in Neuralet infrastructure"
    "Not Reached Yet", "The job's workflow have not been reached to this stage yet"     
    "Unexpected Error", "An internal error has occurred"

**#Step 6:**

Download the trained model whenever the job has been finished. ::

    python3 client.py download_file --job_id JOBID

**What is inside :code:`output.zip` file?**

:code:`train_outputs` : Contains all of the Adaptive Learning files.

:code:`train_outputs/frozen_graph` : Contains all of required files for inference and exporting to the edge devices. Pass this directory to :code:`inference.py` in :code:`x86` devices for running inference on trained model.

:code:`train_outputs/frozen_graph/frozen_inference_graph.pb` : When :code:`QuantizedModel` is :code:`false` in config file this file is inside frozen_graph directory. You can pass this file to the Jetson Exporter to create TensorRT engine.

:code:`train_outputs/frozen_graph/detect.tflite` : When :code:`QuantizedModel` is :code:`true` in config file this file is inside frozen_graph directory. This is the qunatized :code:`tflite` file. You can pass it to EdgeTPU exporter to create an edgetpu compiled tflite file.

:code:`event.out.tfevents` : This is the training log file of Adaptive Learning. You can open this file with :code:`tensorboard` and monitor training progress.



Adaptive Learning Config File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To customize the Adaptive Learning framework based on your needs, you must configure the sample config file on :code:`configs/` directory. There is a brief explanation of each parameter of config files in the following table:

.. csv-table:: a title
    :header: "Parameter", "Options", "Comments"
    :widths: 10, 20, 20


    "Teacher/UploadUUID", "a UUID", "Unique id of uploaded input.zip file."
    "Teacher/VideoFile", "string", "Name of the video you zipped and uploaded."
    "Teacher/Classes", "comma-seperated string without space", "A list of classes names that you want to train on. these classes should be a subset of COCO classes. For all COCO classes just put :code:`coco`"
    "Teacher/PostProcessing", "One of :code:`'background_filter'` or :code:`' '` ", "Background filter will apply a background subtraction algorithm on video frames and discards the bounding boxes in which their background pixels rate is higher than a defined threshold."
    "Teacher/ImageFeature", "One of the :code:`'foreground_mask'`, :code:`'optical_flow_magnitude'`, :code:`'foreground_mask && optical_flow_magnitude'` or :code:`' '`", "This parameter specifies the type of input feature engineering that will perform for training. :code:`'foreground_mask'` replaces one of the RGB channels with the foreground mask. :code:`'optical_flow_magnitude'` replaces one of the RGB channels with the magnitude of optical flow vectors and, :code:`'foreground_mask && optical_flow_magnitude'` performs two feature engineering technique at the same time as well as changing the remaining RGB channel with the grayscale transformation of the frame. For more information about feature engineering and its impact on the model's accuracy, visit `our blog <https://neuralet.com/article/adaptive-learning/>`_ ."
    "Student/QuantizedModel", "true or false", "whether to train the student model with quantization aware strategy or not. This is especially useful when you want to deploy the final model on an edge device that only supports :code:`Int8` precision like Edge TPU. By applying quantization aware training the App will export a :code:`tflite` too."

