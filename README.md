# TensorFlow Object Detection API Tutorial

This repository has the code from my O'Reilly article '' published on.


## Required Packages
* [TensorFlow v1.3](http://www.tensorflow.org/)
* [Jupyter](http://jupyter.org/)
* [NumPy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)



There are two ways you can install these packages: by using Docker or by using native Python 3.5.  If you use Docker you will not be able to render the environment.  

### Using Docker

1. Download and install [Docker](https://www.docker.com/).  If using Ubuntu 14.04/16.04 I wrote my own instructions for installing docker [here](https://github.com/wagonhelm/ML-Workstation-Installation-Guide#install-docker).

2. Download and unzip [this entire repo from GitHub](https://github.com/wagonhelm/TF_ObjectDetection_API), either interactively, or by entering
    ```bash
    git clone https://github.com/wagonhelm/TF_ObjectDetection_API.git

    ```

3. Open your terminal and use `cd` to navigate into the directory of the repo on your machine
    ```bash
    cd TF_ObjectDetection_API
    ```
    
4. To build the Dockerfile, enter
    ```bash
    docker build -t object_dockerfile -f dockerfile .
    ```
    If you get a permissions error on running this command, you may need to run it with `sudo`:
    ```bash
    sudo docker build -t object_dockerfile -f dockerfile .
    ```

5. Run Docker from the Dockerfile you've just built
    ```bash
    docker run -it -p 8888:8888 -p 6006:6006 object_dockerfile bash
    ```
    or
    ```bash
    sudo docker run -it -p 8888:8888 -p 6006:6006 object_dockerfile bash
    ```
    if you run into permission problems.

6. Launch Jupyter and Tensorboard both by using tmux 
    ```bash
    tmux
    
    jupyter notebook --allow-root
    ```
    `Press CTL+B` then `C` to open a new tmux window, then
    
    ```bash
    tensorboard --logdir='/tmp/dpg'
    ```
    To switch windows `Press CTL+B` then `window #` 
 
    Once both jupyter and tensorboard are running, using your browser, navigate to the URLs shown in the terminal output if those don't work  try http://localhost:8888/ for Jupyter Notebook and http://localhost:6006/ for Tensorboard.
    
### Using Native Python 3

1. Install system requirements

```bash
sudo apt-get install git cmake zlib1g-delibjpeg-dev libboost-all-dev libsdl2-dev wget unzip gitboostbuild-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev python3-pip python3-dev python3-wheel
```
2. Download and unzip [this entire repo from GitHub](https://github.com/wagonhelm/TF_ObjectDetection_API), either interactively, or by entering

```bash
git clone https://github.com/wagonhelm/TF_ObjectDetection_API.git
```

3. Install Python Requirement

```bash
cd TF_ObjectDetection_API
pip3 install -r requirements.txt
pip3 install tensorflow jupyter
```

4. Launch Jupyter
```bash
jupyter notebook
```
5. Launch Tensorboard
```bash
tensorboard --logdir='/tmp/dpg'
```
Once both jupyter and tensorboard are running, using your browser, navigate to the URLs shown in the terminal output if those don't work  try http://localhost:8888/ for Jupyter Notebook and http://localhost:6006/ for Tensorboard.


    
