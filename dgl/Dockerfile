FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN mkdir /logs/
RUN mkdir /dataset/
RUN mkdir /checkpoints/

COPY ./dgl/ /dgl/
COPY ./requirements.txt /

WORKDIR /dgl/

RUN pip install --upgrade pip
RUN pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install dgl -f https://data.dgl.ai/wheels/cu113/repo.html
RUN pip install -r /requirements.txt
RUN pip uninstall -y outdated