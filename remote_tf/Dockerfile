FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git\
    && rm -rf /var/lib/apt/lists/

RUN pip install keras matplotlib transformers tensorflow_datasets

RUN pip install -q git+https://github.com/tensorflow/examples.git
