FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

COPY ./degree_generator.py /
COPY ./degree_generator.sh /
COPY ./requirements.txt /

RUN mkdir /logs/
RUN mkdir -p /dataset/reddit/

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    pip install -r /requirements.txt

CMD ["bash", "/degree_generator.sh"]
