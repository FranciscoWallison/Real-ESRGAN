#!/bin/bash
echo "============================================"
echo "  Real-ESRGAN - Teste de GPU e Ambiente"
echo "============================================"
echo ""

echo "=== nvidia-smi ==="
nvidia-smi || echo "AVISO: nvidia-smi nao disponivel (driver fora do container?)"
echo ""

echo "=== PyTorch + CUDA ==="
python3 -c "
import torch

print('PyTorch version :', torch.__version__)
print('CUDA compilado  :', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU count       :', torch.cuda.device_count())
    print('GPU name        :', torch.cuda.get_device_name(0))
    print('CUDA version    :', torch.version.cuda)
    t = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print('Tensor na GPU   :', t.device, '->', t)
    print('STATUS: GPU OK')
else:
    print('AVISO: CUDA nao disponivel - rodando em CPU')
    print('Verifique: nvidia-container-toolkit instalado? docker run com --gpus all?')
"
echo ""

echo "=== Verificação do servidor (se estiver rodando) ==="
if curl -sf http://localhost:8000/health > /tmp/health_response.json 2>/dev/null; then
    echo "Servidor respondeu:"
    python3 -m json.tool /tmp/health_response.json
else
    echo "Servidor nao esta rodando em localhost:8000"
    echo "Para iniciar: docker compose up -d"
fi
echo ""

echo "=== Pesos disponíveis ==="
WEIGHTS_DIR=${WEIGHTS_DIR:-/app/weights}
echo "Diretório: $WEIGHTS_DIR"
ls -lh "$WEIGHTS_DIR"/*.pth 2>/dev/null || echo "Nenhum peso (.pth) encontrado em $WEIGHTS_DIR"
echo ""

echo "============================================"
echo "  Teste concluido"
echo "============================================"
