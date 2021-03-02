Deploy Model Using NVIDIA DeepStream
====================================

After downloading your trained student `ssd_mobilenet_v2` model, you can deploy it by using DeepStream. 

You need to copy your `frozen_inference_graph` which is located in this
`train_outputs/frozen_graph/frozen_inference_graph.pb`
path to deepstream-data directory. ::

    cp train_outputs/frozen_graph/frozen_inference_graph.pb deepstream-data/ 

Then you need to generate and put a `label.txt` file to the `ssd_mobilenet_v2_coco` directory. If you have trained your model with all of the coco labels, you can skip this step as the label file will be downloaded automatically. The label.txt file consists of trained model labels, each in one line without any sign or number. 

There are three other configuration files as follows; `config.pbtxt` is prepared in `ssd_mobilenet_v2_coco` directory and contains name, shape, and size of the network’s output nodes of the model. These vary depending on the model and must change if you use a different model. Some tools that you can use to explore your model and learn about its inputs and outputs are `Netron` and `TensorBoard`.

`source1_primary_ssd_mobilenet_v2_coco.txt` is the DeepStream main configuration file and consists of properties of components in the Deepstream pipeline. You need to set your video path in `uri` key of the `source0` section and your output video path in `output-file` property in `sink0` section. 

`Config_infer_primary_ssd_mobilenet_v2_coco.txt` is the Deepstream inference configuration file and consists of the inference plugin information regarding the inputs, outputs, preprocessing, post-processing, and communication facilities required by the application. If you need to change `label.txt` path, you can edit `labelfile_path` in `postprocess` section in this file.  

After setting all above-mentioned properties, you can use the following commands to run Neuralet docker container for running inference using the DeepStream: ::

    docker build -f deepstream-x86.Dockerfile -t "neuralet/object-detection:deepstream-x86" .
    docker run -it --gpus all  --runtime nvidia -v "$PWD":/repo  neuralet/object-detection:deepstream-x86
