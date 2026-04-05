#!/bin/bash
set -e

WEIGHTS_DIR=${WEIGHTS_DIR:-/app/weights}
MODEL_NAME=${MODEL_NAME:-RealESRGAN_x4plus}

echo "[entrypoint] MODEL_NAME=$MODEL_NAME"
echo "[entrypoint] WEIGHTS_DIR=$WEIGHTS_DIR"

download_if_missing() {
    local filename="$1"
    local url="$2"
    if [ ! -f "$WEIGHTS_DIR/$filename" ]; then
        echo "[entrypoint] Baixando $filename..."
        wget -q --show-progress "$url" -P "$WEIGHTS_DIR"
        echo "[entrypoint] $filename baixado."
    else
        echo "[entrypoint] $filename ja existe, pulando."
    fi
}

# Modelo padrão: sempre garante que está disponível
download_if_missing \
    "RealESRGAN_x4plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Modelo para anime e pixel art
download_if_missing \
    "RealESRGAN_x4plus_anime_6B.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

# Modelo 2x para upscaling mais conservador
download_if_missing \
    "RealESRGAN_x2plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

echo "[entrypoint] Iniciando servidor FastAPI na porta 8000..."
exec python3 server.py
