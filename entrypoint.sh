#!/bin/bash
set -e

WEIGHTS_DIR=${WEIGHTS_DIR:-/app/weights}
MODEL_NAME=${MODEL_NAME:-RealESRGAN_x4plus}

echo "[entrypoint] MODEL_NAME=$MODEL_NAME"
echo "[entrypoint] WEIGHTS_DIR=$WEIGHTS_DIR"

# Download peso padrão se não existir
if [ ! -f "$WEIGHTS_DIR/RealESRGAN_x4plus.pth" ]; then
    echo "[entrypoint] Baixando RealESRGAN_x4plus.pth..."
    wget -q --show-progress \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -P "$WEIGHTS_DIR"
    echo "[entrypoint] Download concluido."
else
    echo "[entrypoint] Peso RealESRGAN_x4plus.pth ja existe, pulando download."
fi

echo "[entrypoint] Iniciando servidor FastAPI na porta 8000..."
exec python3 server.py
