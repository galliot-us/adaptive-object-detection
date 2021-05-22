FROM nvcr.io/nvidia/deepstream-l4t:5.1-21.02-samples 

RUN apt-get update && apt install python3-gi python3-dev python3-gst-1.0 python3-numpy -y
RUN apt-get update && apt-get install gir1.2-gst-rtsp-server-1.0 -y

VOLUME  /repo
WORKDIR /repo/deepstream/5.1

ENTRYPOINT ["bash", "deepstream-51-x86.bash"]
