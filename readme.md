#requirements

numpy

pip3 install git+git://github.com/waspinator/pycococreator.git@0.2.0

#docker detectron2 is necessary to run training
sudo docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -p 8877:8888 -v ~/glock:/glock -v ~/Pobrane:/Pobrane detectron2:v0

#example how to use weight and biases with Detectron2
https://github.com/wandb/artifacts-examples/tree/master/detectron2
