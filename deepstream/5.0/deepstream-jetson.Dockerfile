FROM nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-samples

RUN apt update && apt install -y wget g++ python3-pip
RUN pip3 install --upgrade google-api-python-client


VOLUME  /repo
WORKDIR /repo/deepstream/5.0/
ENTRYPOINT ["bash", "deepstream-jetson.bash"]

