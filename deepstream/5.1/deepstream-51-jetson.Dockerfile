FROM nvcr.io/nvidia/deepstream-l4t:5.1-21.02-samples 

RUN apt-get update && apt install python3-gi python3-dev python3-gst-1.0 python3-numpy python3-pip -y
RUN apt-get update && apt-get install gir1.2-gst-rtsp-server-1.0 -y
RUN pip3 install --upgrade google-api-python-client

VOLUME  /repo
WORKDIR /repo/deepstream/5.1

ENTRYPOINT ["bash", "deepstream-51-jetson.bash"]
