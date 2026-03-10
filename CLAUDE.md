# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

pip install -e .

# Optional: reduces GPU memory usage
# On machines with <96GB RAM: MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
pip install -U flash-attn --no-build-isolation
```

Key pinned dependencies: `transformers==4.57.3`, `accelerate==1.12.0`.

## Downloading models

```bash
# ModelScope (recommended for Mainland China)
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir ./Qwen3-TTS-Tokenizer-12Hz

# Hugging Face
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
```

## Running examples

```bash
python examples/test_model_12hz_base.py          # Voice clone (Base model)
python examples/test_model_12hz_custom_voice.py  # CustomVoice model
python examples/test_model_12hz_voice_design.py  # VoiceDesign model
python examples/test_tokenizer_12hz.py           # Tokenizer encode/decode only
```

## Local demo server (custom)

This repo has a custom FastAPI server (`tts_server.py`) and frontend (`static/index.html`) beyond the upstream `qwen-tts-demo` CLI. It is managed as a systemd service:

```bash
sudo systemctl start tts.service
sudo systemctl restart tts.service
sudo systemctl status tts.service
```

The server runs on port 8001, served at `/tts/` via a reverse proxy. API routes are prefixed `/tts/api/`. Static files are served from `static/`. Generated audio is saved to `output_audio/`. Saved voice clone prompts (`.pt` files) are stored in `saved_voices/`.

Key server behaviors:
- Only one model is loaded at a time; switching model type unloads the previous one.
- `static/index.html` is served with `Cache-Control: no-cache` — no need to restart the service after editing it.
- Restarting IS required when changing `tts_server.py`.

## Fine-tuning workflow

```bash
cd finetuning

# Step 1: pre-extract audio codes from raw JSONL
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# Step 2: SFT
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name my_speaker
```

## Released models

| Model | Instruction Control | Notes |
|-------|-------------------|-------|
| Qwen3-TTS-12Hz-1.7B-Base | — | Voice clone, fine-tunable |
| Qwen3-TTS-12Hz-0.6B-Base | — | Lightweight version |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | ✅ | 9 built-in speakers |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | — | 9 built-in speakers, no instruct |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | ✅ | Natural-language voice design |

Built-in CustomVoice speakers: `Vivian` (zh), `Serena` (zh), `Uncle_Fu` (zh), `Dylan` (zh-Beijing), `Eric` (zh-Sichuan), `Ryan` (en), `Aiden` (en), `Ono_Anna` (ja), `Sohee` (ko).

## Package structure

```
qwen_tts/
  __init__.py                        # exports Qwen3TTSModel, VoiceClonePromptItem, Qwen3TTSTokenizer
  inference/
    qwen3_tts_model.py               # Qwen3TTSModel wrapper + VoiceClonePromptItem dataclass
    qwen3_tts_tokenizer.py           # standalone tokenizer wrapper
  core/
    models/
      modeling_qwen3_tts.py          # Qwen3TTSForConditionalGeneration (full model)
      configuration_qwen3_tts.py     # Qwen3TTSConfig
      processing_qwen3_tts.py        # Qwen3TTSProcessor (text tokenization + collation)
    tokenizer_12hz/                  # 12Hz RVQ codec tokenizer (current)
    tokenizer_25hz/                  # 25Hz tokenizer (legacy)
  cli/demo.py                        # qwen-tts-demo entry point (Gradio UI)
```

The public API surface is `Qwen3TTSModel` (in `inference/`). The `core/` layer is the raw HuggingFace model and should only be touched when modifying model internals.

## Architecture overview

The model stack has three nested levels:

```
Qwen3TTSForConditionalGeneration          (qwen_tts/core/models/modeling_qwen3_tts.py)
  ├── Qwen3TTSTalkerForConditionalGeneration   ← main autoregressive decoder
  │     ├── Qwen3TTSTalkerModel                ← transformer backbone (mRoPE, GQA)
  │     │     ├── codec_embedding              ← codec token embedding table
  │     │     └── text_embedding               ← text token embedding table
  │     ├── text_projection                    ← MLP projecting text→codec dim
  │     ├── codec_head                         ← predicts codec[0] (first codebook)
  │     └── code_predictor                     ← Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
  │           └── predicts codec[1..15]        ← given talker hidden + codec[0]
  ├── Qwen3TTSSpeakerEncoder (Base model only) ← ECAPA-TDNN, 24kHz mel → speaker embedding
  └── speech_tokenizer                         ← loaded from speech_tokenizer/ subfolder
```

### Input embedding construction

The Talker receives a combined embedding at each sequence position: `text_embed + codec_embed`. These two streams are added together (not concatenated), forming a dual-channel sequence. Speaker embedding is injected at a fixed slot (position 6 in the collated sequence) before codec BOS.

### Two-pass codec generation

At each generation step, the Talker predicts `codec[0]` via `codec_head`. The CodePredictor then autoregressively predicts `codec[1..15]` conditioned on the Talker's hidden state + `codec[0]`. Both have independent sampling parameters (`temperature`, `top_k`, `top_p` with `subtalker_*` prefix for CodePredictor).

### Three inference modes

- **Base / Voice Clone**: `tts_model_type="base"`. Requires `ref_audio`. Two sub-modes:
  - ICL mode (`x_vector_only_mode=False`): ref audio codes prepended to context; `ref_text` required.
  - x-vector mode (`x_vector_only_mode=True`): only speaker embedding injected; no `ref_text` needed.
- **CustomVoice**: `tts_model_type="custom_voice"`. Speaker looked up from `config.talker_config.spk_id` dict → codec_embedding row. Speaker slot `3000` is reserved for fine-tuned speakers.
- **VoiceDesign**: `tts_model_type="voice_design"`. No speaker embedding; natural-language `instruct` text prepended via `text_projection`.

### Model type is determined by `config.json`

`tts_model_type` in `config.json` controls which inference path is active. Fine-tuning converts `base` → `custom_voice` by rewriting `config.json` and embedding the target speaker into `codec_embedding.weight[3000]`.

### Speech tokenizer lives separately

The speech tokenizer (`Qwen3TTSTokenizer`) is loaded from `speech_tokenizer/` inside the model directory. `Qwen3TTSForConditionalGeneration.from_pretrained` handles downloading and loading it automatically. The tokenizer operates on 12Hz codec tokens (RVQ with 32 codebooks, `num_code_groups=32`).

### Fine-tuning specifics

- `finetuning/dataset.py` `collate_fn` builds the dual-channel `input_ids` tensor of shape `(B, T, 2)` manually; understanding the position layout (prefix tokens, speaker slot at position 6, BOS/EOS positions) is essential when modifying the data pipeline.
- Only the Base model supports fine-tuning via `sft_12hz.py`. The speaker encoder weights are dropped when saving checkpoints (not needed at inference).
- Loss = main talker CE loss on `codec[0]` + `0.3 × sub-talker loss` on `codec[1..15]`.
- Reference audio for fine-tuning **must be 24kHz** (asserted in `extract_mels`).

## Advanced usage: Voice Design then Clone

Design once, clone many times:

```python
# 1. Use VoiceDesign to synthesize a reference clip
ref_wavs, sr = design_model.generate_voice_design(text=ref_text, instruct=ref_instruct)

# 2. Extract reusable clone prompt from that clip (computed once)
prompt = clone_model.create_voice_clone_prompt(ref_audio=(ref_wavs[0], sr), ref_text=ref_text)

# 3. Reuse prompt across multiple generations without re-extracting features
wavs, sr = clone_model.generate_voice_clone(text="Sentence A", voice_clone_prompt=prompt)
wavs, sr = clone_model.generate_voice_clone(text="Sentence B", voice_clone_prompt=prompt)
```
