# FROM ubuntu:18.04

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

LABEL maintainer="Xiaonan Zhao"
LABEL version="1.0"
LABEL description="docker image with Pytorch"

# LABEL multi.label1="yolov4" multi.label2="API" other="cpu-only"

WORKDIR /app

# copy all files under root dir yolov4/ to root dir
COPY . /app

RUN pip install -r requirements-train.txt

RUN cd /app

ENTRYPOINT [ "python", "train.py"]
