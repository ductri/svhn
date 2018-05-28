FROM nvidia/cuda
MAINTAINER Duc Tri trind@younetgroup.com
ENV REFRESHED_AT 2018-05-28
RUN apt-get -qq update

RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install matplotlib==2.1.2
RUN pip3 install tensorflow-gpu

RUN mkdir /source
ADD ./ /source/

WORKDIR /source

CMD python3 train.py
