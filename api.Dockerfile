FROM tensorflow/tensorflow:2.2.2-py3

VOLUME  /repo
WORKDIR /repo

COPY ./requirements.txt /repo/requirements.txt

RUN apt update && apt install -y libgl1-mesa-glx zip 
RUN pip install -r requirements.txt

ENV HOST="0.0.0.0"
ENV PORT=5000

ENTRYPOINT uvicorn api:app --host ${HOST} --port ${PORT}

