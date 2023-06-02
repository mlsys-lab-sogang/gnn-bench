FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN mkdir /logs/
RUN mkdir /checkpoints/

COPY ./dgl/ /dgl/
COPY ./requirements.txt /

# To use ssh inside container.
RUN apt-get update
RUN apt-get install -y openssh-server

RUN ["/bin/bash", "-i", "-c", "ssh-keygen -f ~/.ssh/id_rsa -N ''"]
RUN ["/bin/bash", "-i", "-c", "cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys"]

RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

ENTRYPOINT service ssh restart && service sshd restart && bash

WORKDIR /dgl/

RUN pip install --upgrade pip
RUN pip install dgl -f https://data.dgl.ai/wheels/cu113/repo.html
RUN pip install -r /requirements.txt
RUN pip uninstall -y outdated

# CMD ["/bin/bash", "partition_graph.sh"]