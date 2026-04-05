import io
import os
import sys
import types
import glob

# Shim de compatibilidade: torchvision >= 0.16 removeu functional_tensor
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    import torchvision.transforms.functional as _tvf
    _mod = types.ModuleType("torchvision.transforms.functional_tensor")
    _mod.rgb_to_grayscale = _tvf.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _mod

import cv2
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

DEFAULT_MODEL = os.getenv("MODEL_NAME", "RealESRGAN_x4plus")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/app/weights")

# Modelos conhecidos com configuração completa
KNOWN_MODELS = {
    "RealESRGAN_x4plus": {
        "arch": "rrdb", "num_block": 23, "scale": 4,
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "description": "Fotos reais, uso geral — 4x",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "arch": "rrdb", "num_block": 6, "scale": 4,
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "description": "Anime, ilustrações e pixel art — 4x",
    },
    "RealESRGAN_x2plus": {
        "arch": "rrdb", "num_block": 23, "scale": 2,
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "description": "Fotos reais em 2x",
    },
    "realesr-general-x4v3": {
        "arch": "srvgg", "num_conv": 32, "scale": 4,
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "description": "Geral com denoising — 4x",
    },
}


def detect_model_from_state_dict(state_dict: dict) -> dict:
    """Detecta arquitetura do modelo a partir do state dict (para .pth customizados)."""
    # SRVGGNet tem chaves específicas
    if any("body.0.weight" in k for k in state_dict.keys()):
        return {"arch": "srvgg", "num_conv": 32, "scale": 4}

    # Conta blocos RRDB para detectar num_block
    max_block = 0
    for k in state_dict.keys():
        if "RRDB_trunk." in k or "body." in k:
            try:
                idx = int(k.split(".")[1])
                max_block = max(max_block, idx)
            except (ValueError, IndexError):
                pass

    num_block = max_block + 1 if max_block > 0 else 23

    # Detecta scale pelo tamanho das camadas de upsampling
    scale = 4
    upscale_keys = [k for k in state_dict if "upconv" in k or "upsample" in k]
    if len(upscale_keys) == 1:
        scale = 2

    return {"arch": "rrdb", "num_block": num_block, "scale": scale}


def build_upsampler(model_name: str) -> RealESRGANer:
    """Inicializa o RealESRGANer para um modelo conhecido ou .pth customizado."""

    if model_name in KNOWN_MODELS:
        cfg = KNOWN_MODELS[model_name]
        model_path = os.path.join(WEIGHTS_DIR, cfg["filename"])

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Peso '{cfg['filename']}' não encontrado em {WEIGHTS_DIR}. "
                f"Baixe com: wget {cfg['url']} -P {WEIGHTS_DIR}"
            )

        if cfg["arch"] == "srvgg":
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=cfg["num_conv"], upscale=cfg["scale"], act_type="prelu"
            )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"]
            )
        scale = cfg["scale"]

    else:
        # Modelo customizado: qualquer .pth na pasta weights/
        model_path = os.path.join(WEIGHTS_DIR, model_name)
        if not model_path.endswith(".pth"):
            model_path += ".pth"

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Modelo '{model_name}' não encontrado em {WEIGHTS_DIR}. "
                f"Coloque o arquivo .pth na pasta weights/ e tente novamente."
            )

        # Detecta arquitetura a partir do state dict
        loadnet = torch.load(model_path, map_location="cpu")
        if "params_ema" in loadnet:
            state_dict = loadnet["params_ema"]
        elif "params" in loadnet:
            state_dict = loadnet["params"]
        else:
            state_dict = loadnet

        detected = detect_model_from_state_dict(state_dict)
        scale = detected["scale"]

        if detected["arch"] == "srvgg":
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=detected["num_conv"], upscale=scale, act_type="prelu"
            )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=detected["num_block"], num_grow_ch=32, scale=scale
            )

        print(f"[server] Modelo customizado '{model_name}' detectado: {detected}")

    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )


def list_available_models() -> list[dict]:
    """Lista modelos conhecidos + .pth customizados na pasta weights/."""
    result = []

    for name, cfg in KNOWN_MODELS.items():
        path = os.path.join(WEIGHTS_DIR, cfg["filename"])
        result.append({
            "name": name,
            "filename": cfg["filename"],
            "description": cfg["description"],
            "scale": cfg["scale"],
            "available": os.path.isfile(path),
            "download_url": cfg["url"],
        })

    # Adiciona .pth customizados que não são modelos conhecidos
    known_files = {cfg["filename"] for cfg in KNOWN_MODELS.values()}
    for pth_path in glob.glob(os.path.join(WEIGHTS_DIR, "*.pth")):
        filename = os.path.basename(pth_path)
        if filename not in known_files:
            result.append({
                "name": filename.replace(".pth", ""),
                "filename": filename,
                "description": "Modelo customizado",
                "scale": "auto-detectado",
                "available": True,
                "download_url": None,
            })

    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cache de upsamplers: carrega o modelo padrão na inicialização
    app.state.upsamplers = {}
    print(f"[server] Carregando modelo padrão '{DEFAULT_MODEL}'...")
    print(f"[server] CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[server] GPU: {torch.cuda.get_device_name(0)}")
    app.state.upsamplers[DEFAULT_MODEL] = build_upsampler(DEFAULT_MODEL)
    print(f"[server] Pronto. Modelos disponíveis: {[m['name'] for m in list_available_models() if m['available']]}")
    yield
    app.state.upsamplers.clear()


app = FastAPI(
    title="Real-ESRGAN API",
    description="API para upscaling de imagens com suporte a múltiplos modelos e GPU",
    version="2.0.0",
    lifespan=lifespan,
)


def get_upsampler(model_name: str) -> RealESRGANer:
    """Retorna upsampler do cache ou carrega sob demanda."""
    if model_name not in app.state.upsamplers:
        print(f"[server] Carregando modelo '{model_name}' sob demanda...")
        app.state.upsamplers[model_name] = build_upsampler(model_name)
        print(f"[server] Modelo '{model_name}' carregado.")
    return app.state.upsamplers[model_name]


@app.get("/health")
def health():
    """Status do serviço e GPU."""
    cuda_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(app.state.upsamplers.keys()),
        "cuda": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "device": "cuda:0" if cuda_available else "cpu",
    }


@app.get("/models")
def models():
    """Lista todos os modelos disponíveis (conhecidos + customizados na pasta weights/)."""
    return list_available_models()


@app.post("/upscale")
async def upscale(
    file: UploadFile = File(..., description="Imagem de entrada (JPEG, PNG, WebP, BMP)"),
    model: str = Query(default=DEFAULT_MODEL, description="Nome do modelo (ver GET /models)"),
    outscale: float = Query(default=4.0, ge=1.0, le=8.0, description="Fator de escala final (1–8x)"),
    tile: int = Query(default=0, ge=0, description="Tile size para economizar VRAM (0 = desabilitado)"),
):
    """
    Realiza upscaling de uma imagem.

    - **model**: escolha o modelo (padrão: RealESRGAN_x4plus). Ver `/models` para lista completa.
    - **outscale**: fator de ampliação final (ex: 4 = 4x maior)
    - **tile**: para imagens grandes com VRAM limitada (ex: tile=512)

    Exemplos de modelo para pixel art:
    - `RealESRGAN_x4plus_anime_6B` — melhor para sprites e pixel art
    - `4x-UltraSharp` — coloque o .pth em weights/ e use o nome sem extensão
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem inválida. Formatos: JPEG, PNG, WebP, BMP")

    try:
        upsampler = get_upsampler(model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {e}")

    original_tile = upsampler.tile
    if tile > 0:
        upsampler.tile = tile

    try:
        output, img_mode = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "hint": "VRAM insuficiente. Tente ?tile=512 ou ?tile=256",
            }
        )
    finally:
        upsampler.tile = original_tile

    ext = "png" if img_mode == "RGBA" else "jpg"
    _, encoded = cv2.imencode(f".{ext}", output)

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type=f"image/{ext}",
        headers={
            "X-Model": model,
            "X-Outscale": str(outscale),
            "X-CUDA": str(torch.cuda.is_available()),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
