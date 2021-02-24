FROM nvcr.io/nvidia/deepstream:5.0.1-20.09-triton

RUN apt update && apt install -y vim wget

VOLUME  /repo
WORKDIR /repo

ENTRYPOINT ["bash", "deepstream.bash"]
CMD ["-f", "$FROZEN_INFERENCE_GRAPH_PATH"]

