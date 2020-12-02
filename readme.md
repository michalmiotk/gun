#requirements

numpy

pip3 install git+git://github.com/waspinator/pycococreator.git@0.2.0

# DOCKER
dependencies:  
hardware:  nvidia gpu  
software: docker nvidia-docker2 - below are link how to install it:
 https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
### run with docker detectron2 is necessary to run training
```
sudo docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -p 8877:8888 -v ~/weapons:/glock -v ~/Pobrane:/Pobrane detectron2:v0
```
now jupyter notebook should run on http://127.0.0.1:8877


#example how to use weight and biases with Detectron2
https://github.com/wandb/artifacts-examples/tree/master/detectron2
