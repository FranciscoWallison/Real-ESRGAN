FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    git python3-pip python3-setuptools python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx ffmpeg \
    g++ wget unzip curl && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean

# Instala torch/torchvision com CUDA 12.1 do índice oficial PyTorch
# (evita conflito de nvidia-cublas com basicsr)
RUN pip3 install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Instala o restante das dependências (sem torch/torchvision para evitar sobreescrita)
RUN grep -vE '^torch|^torchvision' requirements.txt > /tmp/requirements_no_torch.txt && \
    pip3 install --no-cache-dir -r /tmp/requirements_no_torch.txt && \
    pip3 install --no-cache-dir "fastapi>=0.110.0" "uvicorn[standard]>=0.29.0" python-multipart && \
    pip3 install --no-cache-dir rich && \
    pip3 install .

# Patch basicsr: functional_tensor.rgb_to_grayscale foi removido no torchvision 0.16+
RUN sed -i \
    's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Ninja (necessário para compilações do basicsr)
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip && \
    unzip ninja-linux.zip -d /usr/local/bin/ && \
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force && \
    rm ninja-linux.zip

RUN mkdir -p /app/weights /experiments/pretrained_models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
