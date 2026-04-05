import io
import os
import sys
import types

# Shim de compatibilidade: torchvision >= 0.16 removeu functional_tensor
# basicsr ainda importa rgb_to_grayscale desse módulo antigo
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

MODEL_NAME = os.getenv("MODEL_NAME", "RealESRGAN_x4plus")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/app/weights")


def build_upsampler(model_name: str) -> RealESRGANer:
    """Inicializa o RealESRGANer com o modelo especificado.
    Espelha a lógica de seleção de inference_realesrgan.py.
    """
    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )
        scale = 4
        model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")

    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4
        )
        scale = 4
        model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus_anime_6B.pth")

    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2
        )
        scale = 2
        model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x2plus.pth")

    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=32, upscale=4, act_type="prelu"
        )
        scale = 4
        model_path = os.path.join(WEIGHTS_DIR, "realesr-general-x4v3.pth")

    else:
        raise ValueError(
            f"Modelo desconhecido: '{model_name}'. "
            "Opções: RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B, "
            "RealESRGAN_x2plus, realesr-general-x4v3"
        )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Peso não encontrado: {model_path}. "
            "Monte o volume ./weights:/app/weights ou aguarde o download automático."
        )

    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[server] Carregando modelo '{MODEL_NAME}'...")
    print(f"[server] CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[server] GPU: {torch.cuda.get_device_name(0)}")
    app.state.upsampler = build_upsampler(MODEL_NAME)
    print("[server] Modelo carregado. Pronto para receber requisições.")
    yield


app = FastAPI(
    title="Real-ESRGAN API",
    description="API para upscaling de imagens via Real-ESRGAN com suporte a GPU",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Verifica saúde do serviço e status da GPU."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    device = "unknown"
    if hasattr(app.state, "upsampler") and app.state.upsampler is not None:
        try:
            device = str(next(iter(app.state.upsampler.model.parameters())).device)
        except StopIteration:
            pass

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "cuda": cuda_available,
        "gpu": gpu_name,
        "device": device,
    }


@app.post("/upscale")
async def upscale(
    file: UploadFile = File(..., description="Imagem de entrada (JPEG, PNG, WebP)"),
    outscale: float = Query(default=4.0, ge=1.0, le=8.0, description="Fator de escala (1-8x)"),
    tile: int = Query(default=0, ge=0, description="Tamanho do tile (0 = sem tiles, usar para imagens grandes)"),
):
    """Realiza upscaling de uma imagem usando Real-ESRGAN.

    - **file**: Imagem de entrada (JPEG, PNG, WebP)
    - **outscale**: Fator de multiplicação da resolução (padrão: 4x)
    - **tile**: Divide a imagem em tiles para economizar memória GPU (0 = desabilitado)
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Arquivo vazio recebido")

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível decodificar a imagem. Formatos suportados: JPEG, PNG, WebP"
        )

    # Ajusta tile dinamicamente se solicitado
    upsampler = app.state.upsampler
    original_tile = upsampler.tile
    if tile > 0:
        upsampler.tile = tile

    try:
        output, img_mode = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        upsampler.tile = original_tile
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "hint": "Imagem muito grande para a VRAM. Tente ?tile=512 ou ?tile=256"
            }
        )
    finally:
        upsampler.tile = original_tile

    ext = "png" if img_mode == "RGBA" else "jpg"
    success, encoded = cv2.imencode(f".{ext}", output)
    if not success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem de saída")

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type=f"image/{ext}",
        headers={
            "X-Model": MODEL_NAME,
            "X-Outscale": str(outscale),
            "X-CUDA": str(torch.cuda.is_available()),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
