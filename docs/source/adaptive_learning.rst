Adaptive Learning
=================

Neuralet Adaptive Learning Service is designed to customize the object detection models to the provided datasets and specified environments by customers. For more information, please visit our `blog post <https://neuralet.com/article/adaptive-learning/>`_.

Client
^^^^^^

There are two methods for users to call different endpoints of Adaptive Learning API. The first method uses the :code:`curl`'s command-line tool in which the user should type raw commands, and the other more straightforward way is to use our simple Python Client interface.

For using our client interface, you only need to have python on your machine, clone our repository to your system, and go to the provided directory using the commands below. ::
    
    git clone https://github.com/neuralet/edge-object-detection.git
    cd edge-object-detection/services/adaptive-learning/client

Either you prefer to use curl or our interface, we are going to address a step-by-step guide to run Adaptive Learning using each of the methods in the following.

**Step 0: Authentication**
---------------------------
At the beginning, you need to go to our `API home page <https://api.neuralet.io/>`_ and sign up using your email address and password. By clicking on the sign-up button, a verification link will be sent to your email address. After verifying your account, you can sign in to Neuralet’s Adaptive Edge Vision API. Now you are able to get your token using the provided section and keep it for the next steps.

**Step 1: Preparing Video File**
---------------------------------

After logging in and getting your token, you should upload the video file you want to feed Adaptive Learning. Since our API only accepts **zip files** as the input format, you need to create a compressed zip file from your video file, preferably using the name of :code:`input.zip`. ::

    zip -j input.zip PATH_TO_VIDEO_FILE

**Step 2: Uploading Video**
---------------------------

After creating the :code:`input.zip` file, you are ready to upload it to Neuralet’s Servers. After the uploading process, you will receive a unique id (:code:`UUID`), which will be required later in the config file step.

You can upload your file using one of the following methods:

*Client Code:*

If you are using the client code, you should first save your token (provided to you in step0) as a text file and type the path to this file instead of :code:`TOKEN_PATH` in the python command below. The :code:`FILE_PATH` is the address to the :code:`input.zip` file that you have created in step 1. After running the python command successfully, the unique id (:code:`UUID`) will show up. ::

    python3 client.py --token TOKEN_PATH upload_file --file_path FILE_PATH

*Curl Command:*

If you prefer to work with the curl, you need to follow these two stages to upload your video and get your unique id.

* Get Upload URL:

  In this part, you should only copy your token (provided to you in step 0) and paste it instead of :code:`TOKEN` in the command. ::

      curl -X GET "https://api.neuralet.io/api/v1/file/upload/" -H  "accept: application/json" -H "Authorization: Bearer TOKEN"
  
  .. note::
    This endpoint has 2 more optional parameters:

    1. label: Label indicates that in the future you can distinct the uploads from each other (in current situation only distinction between uploads is their ids which can be difficult to read).
    
    2. video_names (comma-separated names): Video names included in zip file in comma-separated format (like :code:`video1.mp4,video2.mp4`).

    If you intend to pass these 2 parameters your url will be like this: ::
        
        https://api.neuralet.io/api/v1/file/upload/?label=crowded-street&?video_names=elm_street.mp4,helm_street.mp4



After running this command, it will return a json containing two items. The first key is :code:`name`, which has your unique id (:code:`UUID`), and you should keep it for the config step. The second key is :code:`upload_object` which contains 2 keys: :code:`url` and :code:`fields`. 

You can see the result in section below: ::
    
    "upload_object":{
        "url":"https://neuralet-adaptive.s3.amazonaws.com/",
        "fields":{
            "key":"fdef1df4-afdb-11eb-b2c8-9a66e4b8f595/input.zip",
            "AWSAccessKeyId":"ASIA...",
            "x-amz-security-token":"IQoJb3...",
            "policy":"eyJleHBpcmF0aW9uI...",
            "signature":"M3bcjKw..."
        }
    }

* Upload File:
  In this stage, you must put the :code:`upload_object.url` you have copied in the last part in place of UploadURL. For the :code:`FILE_PATH`, you need to input the path to the input.zip file you have created in the first step. ::

      curl "UploadURL" -F "key=UUID/input.zip" -F "AWSAccessKeyId"=ASIA..." -F "x-amz-security-token=IQoJb3..." -F "policy-eyJleHBpcmF0aW9uI..." -F "signature=M3bcjKw..." -F "file=@FILE_PATH"


**Step 3: Configure Your Training**
-----------------------------------

In order to start the adaptive learning process on your uploaded file, you should tune and modify the :code:`sample_config.json` file presented in our `repository <https://github.com/neuralet/edge-object-detection/blob/main/services/adaptive-learning/client/configs/sample_config.json>`_.

There are two mandatory fields in the sample_config file, which you are required to set. First, you must copy-paste your unique id (:code:`UUID` provided to you in the previous step) in front of the :code:`UploadUUID` field. The second field is :code:`VideoFile`, in which you should put your video file’s name against it. (Please pay attention that this is the name of your original video, e.g., :code:`softbio.mp4`) ::

    UploadUUID = Your_Unique_ID
    VideoFile = Your_video_File_Name.mp4

In addition, there are a few more fields presented in the sample config file that you can modify based on your requirements. For example, in the config file's :code:`Classes` field, you can choose between 90 different Object Categories of COCO's dataset by writing your desired classes' name with a comma-separated format to train your model. Notice that the default value (:code:`coco`) will train all of the 90 object categories. You can find the 90 classes of COCO’s dataset in their `original research paper <https://arxiv.org/abs/1405.0312>`_. Furthermore, it is possible to change :code:`QuantizedModel` value for the Student network.

To do this, you need to adjust the sample config file on the :code:`configs/` directory. Thus, we have prepared a brief explanation for each of the config files' parameters and options in the following table. You can also use the sample config file in :code:`configs/sample_config.json`.

.. csv-table:: Config File Fields
    :header: "Parameter", "Options", "Comments"


    "Teacher/UploadUUID", "a UUID", "Unique id of uploaded input.zip file."
    "Teacher/VideoFile", "string", "Name of the video you zipped and uploaded."
    "Teacher/Classes", "comma-seperated string", "A list of class names that you want to train your model on. These classes should be a subset of COCO classes. You can find the COCO’s category names in their original paper. To train on all of the 90 COCO classes, just put :code:`'coco'.`"
    "Student/QuantizedModel", "true or false", "whether to train the student model with quantization aware strategy or not. This is especially useful when you want to deploy the final model on an edge device that only supports :code:`Int8` precision like Edge TPU. By applying quantization aware training the App will export a :code:`tflite` too."


**Step 4: Start a Training Job**
--------------------------------

Up until now, you have uploaded your video file and tuned the config file’s parameters for training. Now you are ready to request to train your adaptive learning model.
At the end of this step, by running the command using either the Client code or :code:`curl`, you will get a **Job id** that you should keep for monitoring your training status in the next steps.

*Client Code:*

As same as the second step, you need to input the path to your token text file instead of :code:`TOKEN_PATH` and the address of your config file in the :code:`CONFIG_FILE` field. ::

    python3 client.py --token TOKEN_PATH train --config_path CONFIG_PATH

*Curl Command:*

Again, similar to the second step, you should copy-paste the token we have provided to you at the beginning instead of :code:`TOKEN`. Additionally, you must give the path to your config file in the :code:`JSON_CONFIGFILE_PATH` field. ::

    curl -X POST "https://api.neuralet.io/api/v1/model/train/" -H "accept: application/json" -H "Content-Type: application/json" -H "Authorization: Bearer TOKEN" -d @JSON_CONFIGFILE_PATH


**Step 5: Get Job Status**
--------------------------

At this moment, your model is training on the Neuralet’s servers that may take from a few hours to a couple of days to finish based on the video length. Meanwhile, if you want to know your model’s status at each moment, you are going to use this command.
In this stage, you can request a job status using the **Job id** generated in the last step to observe the operation progress.

*Client Code:*

Enter the address to your token text file and your Job id, respectively, in the provided :code:`TOKEN_PATH` and :code:`JOBID` fields of the command and run it. ::
    
    python3 client.py --token TOKEN_PATH get_status --job_id JOBID

*Curl Command:*

You only need to repeat the previous step and copy-paste your token in the :code:`TOKEN` field, and input your job id in the given field for :code:`JOB_ID`. ::

    curl -X POST "https://api.neuralet.io/api/v1/model/status/" -H  "accept: application/json" -H  "Content-Type: application/json" -H "Authorization: Bearer TOKEN" -d "{\"job_id\":\"JOB_ID\"}"

By running the command and sending your request to our API, you may get one of the following messages for either the Teacher or Student models each time you request for the status (Overall Status):

.. csv-table:: Status Massages
    :header: "Message", "Description"

    "Allocating Resource", "We are Allocating Resources (e.g., a computing machine) to your job."
    "Building", "We have allocated the resources, and the program is Building an environment (installing the required packages) to start your job."
    "Training", "The Training process has started. An Adaptive Learning Job is Running."
    "Wrapping Up", "Your training is about to finish and is Saving data and completing the job."
    "Finished", "The job has been finished successfully."
    "Failed", "If the process faces an infrastructural or hardware problem such as Neuralet’s server failure, you will see this message."
    "Not Reached Yet", "It usually appears as the student model's status, which means the job's workflow has not reached the student model's training phase yet. I.e., while the teacher model is running, the student model's status will be Not Reached Yet."     
    "Unexpected Error", "An internal error has occurred"

Also you get more specific status such as individual status for Teacher and Student plus their progress on the job.

**Step 6: Download your model**
-------------------------------

Finally, you have reached the final step, and the job has finished successfully. Now you can download your Adaptive Learning’s trained student model. After running one of the below commands based on your preference, you will receive a file named :code:`output.zip` that we will explain the contents in the next section.

*Client Code:*

As you would probably know, you should insert the address to your token file in the :code:`TOKEN_PATH` field and replace your job id with :code:`JOBID`, just like what you did in step five. ::

    python3 client.py --token TOKEN_PATH download_file --job_id JOBID

*Curl Command:*

If you are using the curl, there are two stages here to finally get your output file:

* Get your upload link:

  You only need to act like step five once more for replacing the :code:`TOKEN` and :code:`JOB_ID` fields using the token and job id you have saved before. Running this command will return an :code:`upload_link` which you need in the next part. ::

      curl -X POST "https://api.neuralet.io/api/v1/file/download/" -H "accept: application/json" -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" -d "{\"job_id\":\"JOB_ID\"}"

* Download your file:

  Now by putting the :code:`upload_link` that you have received in the previous step against the provided field and running the command, your output file's download process will start. ::

       wget "upload_link" -O output.zip

**What does the output.zip file contain?**

After extracting the output.zip file in your computer, you will see the main directory of this zip file named :code:`train_outputs`, which contains all of the Adaptive Learning files and directories. Here we will walk through the files and directories inside the :code:`train_ouputs` and present a brief explanation of their contents.

First, we are going to introduce the most important files inside the :code:`train_ouputs`:

:code:`validation_vid.mp4` :

This is a video with a maximum length of 40 seconds, which compares the results of running an SSD-MobileNet-V2 model trained on COCO (Baseline model) and the Adaptive Learning trained (Student) model on a validation set video (Not used in the training process). 


:code:`label_map.pbtxt` :

This :code:`pbtxt` file contains a series of mappings that connects a set of class IDs with the corresponding class names. To run the inference code of this module, you should pass this file to the script to classify each object with the right name.

:code:`events.out.tfevents` :

If you want to monitor and analyze your training process, you can open this file using **TensorBoard** and observe each step of the Adaptive Learning model training process.

So far, we have introduced the most important files in the :code:`train_outputs` directory. Now we are going to explain the contents of the :code:`train_outputs/frozen_graph directory`.

:code:`train_outputs/frozen_graph` :

Actually, this is the main directory of our trained model, which contains all of the required files for inferencing and exporting to the edge devices.

:code:`train_outputs/frozen_graph/frozen_inference_graph.pb` :

For running your model on Jetson, you should pass this file to the export module that we have built for edge object detection. So it will export and create a TensorRT engine for you.

:code:`train_outputs/frozen_graph/detect.tflite` :

If you have had set your :code:`QuantizedModel` as :code:`true` in the config file, this file would be available to you inside the frozen_graph directory.
The importance of this file is for exporting your model to the EdgeTPU. In this case, our EdgeTPU exporter accepts this :code:`detect.tflite` file as an input to create an edgetpu compiled tflite file.

:code:`train_outputs/frozen_graph/saved_model` :

This is the last important directory we are introducing here. The :code:`frozen_graph/saved_model` contains a TensorFlow :code:`saved-model` for inferencing on X86s.


Client Management
^^^^^^^^^^^^^^^^^


**Kill Job**
------------

When your model is training, you can cancel your job.
In this stage, you can request a kill job using the **Job id** generated in the :code:`Step 4: Start a Training Job`.

*Client Code:*

Enter the address to your token text file and your Job id, respectively, in the provided :code:`TOKEN_PATH` and :code:`JOBID` fields of the command and run it. ::
    
    python3 client.py --token TOKEN_PATH kill_job --job_id JOBID

*Curl Command:*

You only need to repeat the previous step and copy-paste your token in the :code:`TOKEN` field, and input your job id in the given field for :code:`JOB_ID`. ::

    curl -X POST "https://api.neuralet.io/api/v1/model/kill/" -H  "accept: application/json" -H  "Content-Type: application/json" -H "Authorization: Bearer TOKEN" -d "{\"job_id\":\"JOB_ID\"}"


**User Jobs**
-------------

Get User jobs list.

*Client Code:*

Enter the address to your token text file. respectively, in the provided :code:`TOKEN_PATH` field of the command and run it. ::
    
    python3 client.py --token TOKEN_PATH user_jobs

*Curl Command:*

You only need to repeat the previous step and copy-paste your token in the :code:`TOKEN` field. ::

    curl "https://api.neuralet.io/api/v1/users/me/jobs/" -H "Authorization: Bearer TOKEN"


*Response:*

.. code-block:: json
    
    "jobs": [
        {
            "job_id": "WcLbF1VOB904wk/aMNsfU1==",
            "created_at": "2021-04-05T21:23:31.815000"
        },
        {
            "job_id": "/3I5rFqL+E4sQyskPTLNWg==",
            "created_at": "2021-03-07T16:49:41.249000"
        }
    ],
    "number_of_pages": 1,
    "current_page": 1
    }


**User Uploads**
----------------

Get User uploads list.

*Client Code:*

Enter the address to your token text file. respectively, in the provided :code:`TOKEN_PATH` field of the command and run it. ::
    
    python3 client.py --token TOKEN_PATH user_uploads

*Curl Command:*

You only need to repeat the previous step and copy-paste your token in the :code:`TOKEN` field. ::

    curl "https://api.neuralet.io/api/v1/users/me/uploads/" -H "Authorization: Bearer TOKEN"


*Response:*

.. code-block:: json
    
    {
    "uploads": [
        {
        "name": "fdef2df4-afdb-11eb-b2c8-9a66efb8f595",
        "label": "crowded-street-number-1",
        "created_at": "2021-05-08T09:01:35.795000",
        "video_names": [
            "video1.mp4",
            "video2.mp4"
        ]
        },
        {
        "name": "5ed05020-afaa-11eb-b7cx-6ec41806e103",
        "label": "",
        "created_at": "2021-05-07T17:33:43.757000",
        "video_names": [
            "video11.mp4",
            "video12.mp4"
        ]
        }
    ],
    "number_of_pages": 1,
    "current_page": 1
    }


**User Info**
-------------

Get User detail info.

*Client Code:*

Enter the address to your token text file. respectively, in the provided :code:`TOKEN_PATH` field of the command and run it. ::
    
    python3 client.py --token TOKEN_PATH user_detail

*Curl Command:*

You only need to repeat the previous step and copy-paste your token in the :code:`TOKEN` field. ::

    curl "https://api.neuralet.io/api/v1/users/me/detail/" -H "Authorization: Bearer TOKEN"


*Response:*

.. code-block:: json
    
    {
        "email": "test@test.com",
        "is_active": true,
        "is_superuser": false,
        "is_verified": true
    }