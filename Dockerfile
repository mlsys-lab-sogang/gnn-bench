FROM python:3.8

COPY ./degree_generator.py /degree_generator.py
COPY ./requirements.txt /requirements.txt

RUN mkdir /logs/
RUN mkdir -p /dataset/reddit/

WORKDIR /

RUN pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    pip install -r /requirements.txt

CMD ["python3 degree_generator.py"]
