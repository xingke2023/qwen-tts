#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS Demo Server
Supports: CustomVoice / VoiceDesign / VoiceClone (with saved voices)
"""
import gc
import io
import json
import logging
import os
import tempfile
import traceback

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-TTS Demo")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "custom_voice": os.path.join(BASE_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "voice_design": os.path.join(BASE_DIR, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "base":         os.path.join(BASE_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}
SAVED_VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_current_model = None
_current_model_type = None


def get_model(model_type: str):
    global _current_model, _current_model_type

    if _current_model_type == model_type and _current_model is not None:
        return _current_model

    if _current_model is not None:
        logger.info(f"Unloading {_current_model_type} ...")
        del _current_model
        _current_model = None
        _current_model_type = None
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Loading {model_type} ...")
    from qwen_tts import Qwen3TTSModel
    _current_model = Qwen3TTSModel.from_pretrained(
        MODEL_PATHS[model_type],
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    _current_model_type = model_type
    logger.info(f"{model_type} ready.")
    return _current_model


AUDIO_OUTPUT_DIR = os.path.join(BASE_DIR, "output_audio")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def wav_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def audio_response(wav: np.ndarray, sr: int, prefix: str = "output") -> StreamingResponse:
    import time
    wav_bytes = wav_to_bytes(wav, sr)
    filename = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    save_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(wav_bytes)
    logger.info(f"Saved audio: {save_path}")
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def convert_to_wav(input_path: str) -> str:
    wav_path = input_path + ".wav"
    ret = os.system(f'ffmpeg -y -i "{input_path}" -ar 24000 -ac 1 "{wav_path}" -loglevel error')
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed to convert {input_path}")
    return wav_path


# ---------------------------------------------------------------------------
# API: status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def status():
    return {
        "loaded_model": _current_model_type,
        "gpu_free_mb": (
            torch.cuda.mem_get_info()[0] // 1024 // 1024
            if torch.cuda.is_available() else None
        ),
    }


# ---------------------------------------------------------------------------
# API: CustomVoice
# ---------------------------------------------------------------------------
CUSTOM_VOICE_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

@app.get("/api/speakers")
def get_speakers():
    return {"speakers": CUSTOM_VOICE_SPEAKERS}


@app.post("/api/custom_voice")
def custom_voice(
    text: str = Form(...),
    speaker: str = Form("Vivian"),
    language: str = Form("Auto"),
    instruct: str = Form(""),
):
    try:
        model = get_model("custom_voice")
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct if instruct.strip() else None,
        )
        return audio_response(wavs[0], sr, prefix="custom_voice")
    except Exception as e:
        logger.error("custom_voice error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: VoiceDesign
# ---------------------------------------------------------------------------
@app.post("/api/voice_design")
def voice_design(
    text: str = Form(...),
    instruct: str = Form(...),
    language: str = Form("Auto"),
):
    try:
        model = get_model("voice_design")
        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
        return audio_response(wavs[0], sr, prefix="voice_design")
    except Exception as e:
        logger.error("voice_design error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# API: VoiceClone
# ---------------------------------------------------------------------------

def _load_audio_to_wav(upload: UploadFile) -> str:
    """Save uploaded file to temp, convert to WAV if needed. Returns wav path."""
    suffix = os.path.splitext(upload.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name
    if suffix.lower() != ".wav":
        wav_path = convert_to_wav(tmp_path)
        os.unlink(tmp_path)
    else:
        wav_path = tmp_path
    return wav_path


def _save_voice_prompt(name: str, prompt_item, ref_text: str):
    """Serialize VoiceClonePromptItem to disk."""
    save_path = os.path.join(SAVED_VOICES_DIR, f"{name}.pt")
    data = {
        "ref_code": prompt_item.ref_code.cpu() if prompt_item.ref_code is not None else None,
        "ref_spk_embedding": prompt_item.ref_spk_embedding.cpu(),
        "x_vector_only_mode": prompt_item.x_vector_only_mode,
        "icl_mode": prompt_item.icl_mode,
        "ref_text": ref_text,
        "name": name,
    }
    torch.save(data, save_path)


def _load_voice_prompt(name: str):
    """Load saved voice and return VoiceClonePromptItem."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    save_path = os.path.join(SAVED_VOICES_DIR, f"{name}.pt")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Saved voice '{name}' not found")
    data = torch.load(save_path, map_location="cpu", weights_only=False)
    return VoiceClonePromptItem(
        ref_code=data["ref_code"],
        ref_spk_embedding=data["ref_spk_embedding"],
        x_vector_only_mode=data["x_vector_only_mode"],
        icl_mode=data["icl_mode"],
        ref_text=data["ref_text"],
    )


@app.get("/api/saved_voices")
def list_saved_voices():
    voices = []
    for fname in sorted(os.listdir(SAVED_VOICES_DIR)):
        if fname.endswith(".pt"):
            name = fname[:-3]
            voices.append(name)
    return {"voices": voices}


@app.post("/api/save_voice")
def save_voice(
    name: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...),
):
    """Extract and save voice clone prompt from reference audio."""
    wav_path = None
    try:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="音色名称不能为空")
        # Sanitize name
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_() ")

        wav_path = _load_audio_to_wav(ref_audio)
        model = get_model("base")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=wav_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        _save_voice_prompt(safe_name, prompt_items[0], ref_text)
        logger.info(f"Saved voice: {safe_name}")
        return {"success": True, "name": safe_name}
    except Exception as e:
        logger.error("save_voice error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.delete("/api/saved_voices/{name}")
def delete_voice(name: str):
    save_path = os.path.join(SAVED_VOICES_DIR, f"{name}.pt")
    if not os.path.exists(save_path):
        raise HTTPException(status_code=404, detail="音色不存在")
    os.unlink(save_path)
    return {"success": True}


@app.post("/api/voice_clone")
def voice_clone(
    text: str = Form(...),
    language: str = Form("Auto"),
    # Option A: use saved voice
    saved_voice: str = Form(""),
    # Option B: upload new audio
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    wav_path = None
    try:
        model = get_model("base")

        if saved_voice.strip():
            # Use pre-saved voice prompt
            prompt_item = _load_voice_prompt(saved_voice.strip())
            prompt_items = [prompt_item]
        else:
            # Upload new audio
            if ref_audio is None:
                raise HTTPException(status_code=400, detail="请上传参考音频或选择已保存音色")
            if not ref_text.strip():
                raise HTTPException(status_code=400, detail="请填写参考文本")
            wav_path = _load_audio_to_wav(ref_audio)
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=wav_path,
                ref_text=ref_text,
                x_vector_only_mode=False,
            )

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
        )
        return audio_response(wavs[0], sr, prefix="voice_clone")
    except Exception as e:
        logger.error("voice_clone error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


# ---------------------------------------------------------------------------
# API: Transcribe (ASR for ref audio auto-fill)
# ---------------------------------------------------------------------------
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        model_path = os.path.expanduser("~/.cache/faster-whisper-small")
        _whisper_model = WhisperModel(model_path, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
        logger.info("Whisper model loaded.")
    return _whisper_model


@app.post("/api/transcribe")
def transcribe(ref_audio: UploadFile = File(...)):
    """Transcribe uploaded audio and return text."""
    tmp_path = None
    wav_path = None
    try:
        suffix = os.path.splitext(ref_audio.filename)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(ref_audio.file.read())
            tmp_path = tmp.name
        if suffix.lower() != ".wav":
            wav_path = convert_to_wav(tmp_path)
        else:
            wav_path = tmp_path
            tmp_path = None  # same file, don't double-delete

        model = get_whisper_model()
        segments, info = model.transcribe(wav_path, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        return {"text": text, "language": info.language}
    except Exception as e:
        logger.error("transcribe error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Static files (frontend) - mount LAST so API routes take priority
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static_files")

from fastapi.responses import FileResponse

@app.get("/")
def index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
