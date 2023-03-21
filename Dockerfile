FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

COPY ./degree_generator.py /
COPY ./requirements.txt /

RUN mkdir /logs/
RUN mkdir -p /dataset/reddit/

RUN pip install --upgrade pip
RUN pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
RUN pip install torch-geometric
RUN pip install -r /requirements.txt
RUN pip uninstall outdated

CMD ["python", "/degree_generator.py"]
