FROM gcr.io/tensorflow/tensorflow:1.3.0-devel-py3
RUN apt-get update && apt-get install -y git-core tmux xvfb cmake zlib1g-dev libjpeg-dev libboost-all-dev libsdl2-dev wget unzip zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev 
RUN git clone https://github.com/wagonhelm/Deep-Policy-Gradient.git /notebooks/object
WORKDIR "/notebooks"
RUN pip install -r ./object/requirements.txt
CMD ["/run_jupyter.sh"]
