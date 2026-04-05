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
Verifica status do serviço, GPU e modelos carregados.

### `GET /models`
Lista todos os modelos disponíveis — modelos conhecidos + qualquer `.pth` colocado em `weights/`.

```bash
curl http://localhost:8000/models
```

### `POST /upscale`
Recebe uma imagem e retorna a versão com upscaling aplicado.

**Parâmetros (query string):**
| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `model` | string | `RealESRGAN_x4plus` | Nome do modelo (ver `/models`) |
| `outscale` | float | `4.0` | Fator de escala final (1–8x) |
| `tile` | int | `0` | Tile size para GPUs com pouca VRAM (`512`, `256`). `0` = desabilitado |

**Exemplos:**
```bash
# Foto geral — 4x
curl -X POST "http://localhost:8000/upscale" \
  -F "file=@foto.jpg" --output foto_4x.jpg

# Pixel art / anime — melhor modelo para sprites
curl -X POST "http://localhost:8000/upscale?model=RealESRGAN_x4plus_anime_6B" \
  -F "file=@sprite.png" --output sprite_4x.png

# Modelo customizado (coloque o .pth em weights/)
curl -X POST "http://localhost:8000/upscale?model=4x-UltraSharp" \
  -F "file=@imagem.png" --output resultado.png

# Imagem grande com pouca VRAM
curl -X POST "http://localhost:8000/upscale?outscale=4&tile=512" \
  -F "file=@imagem_grande.png" --output resultado.png
```

Documentação interativa: `http://localhost:8000/docs`

---

## Modelos disponíveis

Os três modelos principais são baixados automaticamente no startup. O modelo padrão é definido por `MODEL_NAME` no `docker-compose.yml`.

| Modelo | Uso recomendado | Escala | Auto-download |
|---|---|---|---|
| `RealESRGAN_x4plus` | Fotos reais, uso geral *(padrão)* | 4x | ✅ |
| `RealESRGAN_x4plus_anime_6B` | Anime, ilustrações, **pixel art** | 4x | ✅ |
| `RealESRGAN_x2plus` | Fotos reais em 2x | 2x | ✅ |
| `realesr-general-x4v3` | Geral com denoising | 4x | manual |

### Modelos customizados (pixel art e outros)

Qualquer `.pth` da comunidade colocado em `weights/` fica disponível automaticamente via API. A arquitetura é detectada automaticamente.

```bash
# Exemplo: 4x-UltraSharp (ótimo para arte digital e pixel art)
wget "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" -P weights/

# Após colocar o arquivo, já está disponível:
curl -X POST "http://localhost:8000/upscale?model=4x-UltraSharp" \
  -F "file=@sprite.png" --output resultado.png
```

Busque mais modelos em: [OpenModelDB](https://openmodeldb.info) — filtre por `pixel art`, `anime` ou `photography`.

### Qual modelo usar para pixel art de jogos?

| Caso | Modelo recomendado |
|---|---|
| Sprites e pixel art | `RealESRGAN_x4plus_anime_6B` |
| Arte digital / linhas nítidas | `4x-UltraSharp` (download manual) |
| Screenshot de jogo 3D | `RealESRGAN_x4plus` |
| Foto com rosto | `RealESRGAN_x4plus` + CodeFormer |

Os pesos ficam em `./weights/` — o volume garante que sobrevivem entre rebuilds.

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
