# Real-ESRGAN — Docker + GPU + API REST

Fork do [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) com foco em execução via **Docker com GPU NVIDIA** e exposição como **API REST** (FastAPI), pronto para integração com outros serviços.

---

## Requisitos

- Docker Desktop com suporte a GPU (nvidia-container-toolkit)
- NVIDIA GPU com driver compatível com CUDA 12.1
- (Opcional) `docker compose` v2+

---

## Início rápido

```bash
git clone https://github.com/FranciscoWallison/Real-ESRGAN.git
cd Real-ESRGAN

# Build e subir o container
docker compose up -d

# Aguardar o modelo carregar (~30s, ou ~90s na primeira execução — faz download dos pesos)
# Verificar status
curl http://localhost:8000/health
```

Resposta esperada:
```json
{
  "status": "ok",
  "model": "RealESRGAN_x4plus",
  "cuda": true,
  "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
  "device": "cuda:0"
}
```

---

## API

### `GET /health`
Verifica se o serviço está rodando e retorna informações da GPU.

### `POST /upscale`
Recebe uma imagem e retorna a versão com upscaling aplicado.

**Parâmetros (query string):**
| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `outscale` | float | `4.0` | Fator de escala (1–8x) |
| `tile` | int | `0` | Tamanho do tile para GPUs com pouca VRAM (ex: `512`, `256`). `0` = desabilitado |

**Exemplo com curl:**
```bash
curl -X POST "http://localhost:8000/upscale?outscale=4" \
  -F "file=@inputs/minha_imagem.jpg" \
  --output resultado_4x.jpg
```

**Exemplo com tile para imagens grandes:**
```bash
curl -X POST "http://localhost:8000/upscale?outscale=4&tile=512" \
  -F "file=@inputs/imagem_grande.png" \
  --output resultado.png
```

Documentação interativa disponível em: `http://localhost:8000/docs`

---

## Modelos disponíveis

Defina o modelo pelo `MODEL_NAME` no `docker-compose.yml` ou na variável de ambiente.

| MODEL_NAME | Uso recomendado | Escala |
|---|---|---|
| `RealESRGAN_x4plus` | Fotos reais, uso geral *(padrão)* | 4x |
| `RealESRGAN_x4plus_anime_6B` | Ilustrações e anime | 4x |
| `RealESRGAN_x2plus` | Fotos reais em 2x | 2x |
| `realesr-general-x4v3` | Geral com denoising | 4x |

Os pesos são baixados automaticamente na primeira execução para `./weights/`. Monte um volume para reutilizá-los entre rebuilds.

---

## Configuração Docker

### docker-compose.yml

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./weights:/app/weights   # pesos persistidos entre reinicializações
    environment:
      - MODEL_NAME=RealESRGAN_x4plus
      - WEIGHTS_DIR=/app/weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Para trocar o modelo sem rebuild:
```bash
# Parar, alterar MODEL_NAME no docker-compose.yml, reiniciar
docker compose down
MODEL_NAME=RealESRGAN_x4plus_anime_6B docker compose up -d
```

---

## Testar GPU

```bash
docker compose exec app bash /app/test_gpu.sh
```

Saída esperada:
```
=== nvidia-smi ===
...
=== PyTorch + CUDA ===
CUDA available: True
GPU name: NVIDIA GeForce RTX ...
STATUS: GPU OK

=== Verificação do servidor ===
{"status": "ok", "cuda": true, ...}
```

---

## Uso pela linha de comando (sem API)

O script CLI original continua funcionando dentro do container:

```bash
# Imagens gerais
docker compose exec app python3 inference_realesrgan.py \
  -n RealESRGAN_x4plus -i inputs -o results --fp32

# Anime
docker compose exec app python3 inference_realesrgan.py \
  -n RealESRGAN_x4plus_anime_6B -i inputs -o results

# Com melhoria de rosto (GFPGAN)
docker compose exec app python3 inference_realesrgan.py \
  -n RealESRGAN_x4plus -i inputs -o results --face_enhance

# Vídeo
docker compose exec app python3 inference_realesrgan_video.py \
  -i inputs/video.mp4 -o results/video_upscaled.mp4
```

---

## Integração com server-OTA

Este serviço pode ser usado como backend de upscaling para o projeto [server-OTA](../server-OTA). O `server-OTA` expõe `/api/upscale` que faz proxy para este serviço.

Para subir a stack completa:
```bash
cd ../server-OTA
docker compose up -d
# server-OTA em localhost:3000
# Real-ESRGAN em localhost:8000
```

Endpoint via server-OTA:
```bash
curl -X POST "http://localhost:3000/api/upscale?outscale=4" \
  -F "file=@imagem.jpg" \
  --output resultado.jpg
```

---

## Estrutura do projeto

```
Real-ESRGAN/
├── server.py                  # Servidor FastAPI (API REST)
├── entrypoint.sh              # Inicialização do container (download pesos + start)
├── test_gpu.sh                # Script de verificação de GPU
├── Dockerfile                 # Build com CUDA 12.1 + PyTorch
├── docker-compose.yml         # Orquestração com GPU
├── inference_realesrgan.py    # CLI para upscaling de imagens
├── inference_realesrgan_video.py  # CLI para upscaling de vídeos
├── realesrgan/                # Pacote principal
│   ├── utils.py               # Classe RealESRGANer
│   ├── models/                # Modelos de treino
│   └── archs/                 # Arquiteturas (RRDB, SRVGG)
├── weights/                   # Pesos dos modelos (baixados automaticamente)
├── inputs/                    # Imagens de entrada para CLI
├── options/                   # Configurações de treino (YAML)
└── docs/                      # Documentação adicional
```

---

## Notas técnicas

- **Compatibilidade torchvision:** o `basicsr` usa `torchvision.transforms.functional_tensor` que foi removido na versão 0.16+. O `server.py` aplica um shim de compatibilidade automaticamente.
- **FP16:** habilitado automaticamente quando CUDA está disponível (menor uso de VRAM, mais velocidade). Use `--fp32` na CLI para desabilitar.
- **VRAM insuficiente:** use `?tile=512` ou `?tile=256` no endpoint `/upscale` para processar em blocos.

---

## Referência

- Artigo original: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- Repositório upstream: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
