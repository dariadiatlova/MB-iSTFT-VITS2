FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y unzip \
    ffmpeg \
    espeak \
    gcc
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'NCCL_SOCKET_IFNAME=lo' >> ~/.bashrc

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

RUN cd /app/monotonic_align && mkdir monotonic_align && python setup.py build_ext --inplace

ENTRYPOINT [ "bash" ]