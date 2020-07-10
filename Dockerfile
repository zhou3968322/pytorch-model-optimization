FROM nvidia/cuda:10.1-base
MAINTAINER bingchengzhou@foxmail.com
ENV DEBIAN_FRONTEND="noninteractive"
RUN  apt-get update && apt-get install -y libopencv-dev libxrender-dev libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN apt update && apt-get install -y --no-install-recommends curl && \
 curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
 bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b &&\
 rm Miniconda3-latest-Linux-x86_64.sh && \
 apt-get purge --autoremove -y curl && \
 rm -rf /var/lib/apt/lists/*

ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda
RUN conda install pip && conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
RUN pip install tensorflow==2.1.0
RUN pip install Cython==0.28.5
RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

ADD ./requirements.txt /root/requirements.txt
RUN cd /root && pip install -i http://pypi.douban.com/simple -r requirements.txt && rm -rf requirements.txt

RUN apt update && apt-get install -y --no-install-recommends git && \
    cd /root && git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow && \
    pip install -e .  && \
    apt-get purge --autoremove -y git && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install openssh-server && rm -rf /var/lib/apt/lists/*

# install ssh
RUN mkdir -p /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22

ADD ./data /root/workspace/data
ADD ./src /root/workspace/src
ADD ./tests /root/workspace/tests

CMD /usr/sbin/sshd -D







