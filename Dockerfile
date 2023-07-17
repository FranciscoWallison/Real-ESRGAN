FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    git \
    python3-pip python3-setuptools python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx ffmpeg \
    g++ \
    arch \
    wget unzip && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean && \
    pip3 install --no-cache-dir -r requirements.txt && \
    python3 -m pip install rich && \
    pip3 install .  && \
    cd MM-RealSR && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install . && cd ..

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# RUN apt-get remove --purge nvidia*
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN add-apt-repository contrib
# RUN apt-get update
# RUN apt-get -y install cuda

# ENV NVIDIA_CUDA_PPA=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
# ENV NVIDIA_CUDA_PREFERENCES=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# ENV NVIDIA_CUDA_PUBKEY=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# # Add NVIDIA Developers 3rd-Party PPA
# RUN wget ${NVIDIA_CUDA_PREFERENCES} -O /etc/apt/preferences.d/nvidia-cuda
# RUN apt-key adv --fetch-keys ${NVIDIA_CUDA_PUBKEY}
# RUN echo "deb ${NVIDIA_CUDA_PPA} /" |  tee /etc/apt/sources.list.d/nvidia-cuda.list

# # Install development tools
# RUN apt-get update
# RUN apt-get install -y cuda

# Ninja
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip && \
    unzip ninja-linux.zip -d /usr/local/bin/ && \
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P weights  && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth -P weights  && \
    cd MM-RealSR && \
    wget "https://github.com/TencentARC/MM-RealSR/releases/download/v1.0.0/MMRealSRGAN.pth" -P experiments && \
    python3 setup.py develop  && \
    cd .. && \
    python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir torch>=1.13 opencv-python>=4.7 && \
    pip3 install --no-cache-dir basicsr facexlib realesrgan && \
    pip3 install --no-cache-dir gfpgan && \
    python3 setup.py develop


# weights
RUN mkdir -p /experiments/pretrained_models
#RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth \
#        -P experiments/pretrained_models &&\
#    wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth \
#        -P experiments/pretrained_models

# CMD ["python3", "inference_realesrgan.py", "-i", "/inputs", "-o", "/results", "-n", ""]
CMD [ "bash" ]
