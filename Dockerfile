FROM tensorflow/tensorflow:nightly-gpu-py3
MAINTAINER Duc Tri trind@younetgroup.com
ENV REFRESHED_AT 2018-05-28
RUN apt-get -qq update

RUN apt-get install python-matplotlib
RUN apt-get install python3-tk

VOLUME /source/
VOLUME /all_dataset/

WORKDIR /source

