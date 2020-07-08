FROM ubuntu:18.04
MAINTAINER bingchengzhou@foxmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243

ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-1 && \
ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

RUN apt-get update && apt-get install -y libopencv-dev libxrender-dev libsm6 libxext6

RUN apt update && apt-get install -y --no-install-recommends curl && \
 curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
 bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b &&\
 rm Miniconda3-latest-Linux-x86_64.sh && \
 apt-get purge --autoremove -y curl && \
 rm -rf /var/lib/apt/lists/*

ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda config --set show_channel_urls yes

RUN conda install pip && conda install -y pytorch torchvision cudatoolkit=10.2 && conda install tensorflow-gpu
RUN pip install Cython==0.28.5
RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

ADD ./requirements.txt /root/requirements.txt
RUN cd /root && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && rm -rf requirements.txt

RUN apt update && apt-get install -y --no-install-recommends git && \
    cd /root && git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflw && \
    pip install -e . && cd .. && rm -rf onnx-tensorflow && \
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
ADD ./tests /root/workspce/tests

CMD /usr/sbin/sshd -D







