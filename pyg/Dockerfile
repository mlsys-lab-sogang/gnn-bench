FROM python:3.8

COPY ./ /pyg/
COPY ../requirements.txt /requirements.txt

WORKDIR /pyg/

RUN pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    pip install -r /requirements.txt

ENV MASTER_ADDR $MASTER_ADDR
ENV MASTER_PORT $MASTER_PORT

CMD ["./reddit_sage_dist.sh"]
