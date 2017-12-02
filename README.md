# TensorFlow Object Detection API Tutorial

This repository has the code from my [O'Reilly article](https://www.oreilly.com/ideas/object-detection-with-tensorflow)  published on October 25, 2017.


## Required Packages
* [TensorFlow v1.3](http://www.tensorflow.org/)
* [Jupyter](http://jupyter.org/)
* [NumPy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Scikit-Image](http://scikit-image.org/)
* [Pandas](http://pandas.pydata.org/)
* [lxml](http://lxml.de/)
* [protobuf](https://github.com/google/protobuf)

There are two ways you can install these packages: by using Docker or by using native Python 3.5.

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

6. Install TensorFlow Object Detection API
    ```bash
    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    cd ..
    cd ..
    ```

7. Launch Jupyter and Tensorboard both by using tmux 
    ```bash
    tmux
    
    jupyter notebook --allow-root
    ```
    `Press CTL+B` then `C` to open a new tmux window, then
    
    ```bash
    tensorboard --logdir='data'
    ```
    To switch windows `Press CTL+B` then `window #` 
 
    Once both jupyter and tensorboard are running, using your browser, navigate to the URLs shown in the terminal output if those don't work  try http://localhost:8888/ for Jupyter Notebook and http://localhost:6006/ for Tensorboard.  I had issues with using TensorBoard with Firefox when launched from Docker.
    
### Using Native Python 3

1. Install system requirements

```bash
sudo apt-get install -y git-core wget protobuf-compiler 
```
2. Download and unzip [this entire repo from GitHub](https://github.com/wagonhelm/TF_ObjectDetection_API), either interactively, or by entering

```bash
git clone https://github.com/wagonhelm/TF_ObjectDetection_API.git
```

3. Install Python Requirement

```bash
cd TF_ObjectDetection_API
# Requires sudo if not in a virtual environment
pip3 install -r requirements.txt
pip3 install tensorflow jupyter
```
4. Clone TensorFlow Models Into Repository Directory and Install Object Detection API

```bash
cd TF_ObjectDetection_API
git clone https://github.com/tensorflow/models.git
```

You will have to run this command every time you close your terminal unless you add the the path to slim to your `.bashrc` file

```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
cd ..
```

4. Launch Jupyter
```bash
jupyter notebook
```
5. Launch Tensorboard In New Terminal
```bash
tensorboard --logdir='data'
```
Once both jupyter and tensorboard are running, using your browser, navigate to the URLs shown in the terminal output if those don't work  try http://localhost:8888/ for Jupyter Notebook and http://localhost:6006/ for Tensorboard.


    System information

What is the top-level directory of the model you are using:
research/
Have I written custom code (as opposed to using a stock example script provided in TensorFlow):
Yes
OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
Linux Ubuntu 16.04
TensorFlow installed from (source or binary):
No
CUDA/cuDNN version:
8.0/6.0
GPU model and memory:
1080 ti
Exact command to reproduce:
python -m object_detection/metrics/offline_eval_map_corloc --eval_dir=PATH/TO/EVAL_DIR --eval_config_path=PATH/TO/EVAL_CONGIF.pbtxt --input_config_path=PATH/TO/INPUT_CONFIG.pbtxt

You can obtain the TensorFlow version with

python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"
('v1.4.0-rc1-11-g130a514', '1.4.0')

Describe the problem

object_detection_evaluation states that having the field standard_fields.InputDataFields.groundtruth_difficult is optional. However, it checks whether the field exists or not like this:
groundtruth_dict[standard_fields.InputDataFields.groundtruth_difficult].size
For me, I just removed the .size part and the error got away but you may want to do it in a better way.

INFO:tensorflow:Processing file: /PATH/TO/test_detections.tfrecord-00000-of-00001
INFO:tensorflow:Processed 0 images...
Traceback (most recent call last):
File "/anaconda2/lib/python2.7/runpy.py", line 174, in _run_module_as_main
"main", fname, loader, pkg_name)
File "/anaconda2/lib/python2.7/runpy.py", line 72, in _run_code
exec code in run_globals
File "models/research/object_detection/metrics/offline_eval_map_corloc.py", line 173, in 
tf.app.run(main)
File "/anaconda2/lib/python2.7/site-packages/tensorflow/python/platform/app.py", line 48, in run
_sys.exit(main(_sys.argv[:1] + flags_passthrough))
File "models/research/object_detection/metrics/offline_eval_map_corloc.py", line 166, in main
metrics = read_data_and_evaluate(input_config, eval_config)
File "models/research/object_detection/metrics/offline_eval_map_corloc.py", line 124, in read_data_and_evaluate
decoded_dict)
File "object_detection/utils/object_detection_evaluation.py", line 174, in add_single_ground_truth_image_info
(groundtruth_dict[standard_fields.InputDataFields.groundtruth_difficult]
AttributeError: 'NoneType' object has no attribute 'size'
