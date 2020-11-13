FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip install torch torchvision
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
RUN apt install -y wget
ADD app /app
ADD ../config.yml /config.yml
ADD ../model.pth /model.pth 
RUN wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl -O /model_final_280758.pkl
RUN pip install -r /app/requirements.txt
EXPOSE 80
# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /app
ENTRYPOINT export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && streamlit run GunDetector.py --server.port 80
