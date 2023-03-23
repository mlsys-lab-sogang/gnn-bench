FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN mkdir /logs/
RUN mkdir /dataset/

COPY ./pyg/ /pyg/
COPY ./requirements.txt /

WORKDIR /pyg/

RUN pip install --upgrade pip
RUN pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
RUN pip install -r /requirements.txt
RUN pip uninstall -y outdated

CMD ["bash reddit_sage_dist.sh"]
