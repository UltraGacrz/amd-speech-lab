# AMD Speech Lab 🎙️🔥

**Speech Recognition & TTS on AMD GPUs**

Okay so here's the deal — I got a Radeon VII (yes, the Vega beast with 16GB HBM2) and I was like, why is nobody running Whisper on this thing?? Turns out you totally can with ROCm. This repo is my experiment lab for doing speech stuff on AMD hardware.

## What's in here

- **Whisper fine-tuning on ROCm** — yep, it works. Takes some fiddling but it works.
- **Speech-to-text pipeline** — drop in audio, get text. Simple.
- **Text-to-speech** — using Coqui TTS, sounds pretty decent
- **Speaker diarization** — figure out who's talking when
- **Audio augmentation** — pitch shift, time stretch, noise injection, the usual

## Quick Start

```bash
# Clone it
git clone https://github.com/UltraGacrz/amd-speech-lab.git
cd amd-speech-lab

# Install deps
pip install -r requirements.txt

# Transcribe something
python scripts/transcribe.py --input audio.wav --model openai/whisper-base

# Fine-tune Whisper on your own data
python src/whisper_finetune.py --config configs/whisper_finetune.yaml
```

## ROCm Notes

If you're on ROCm 5.x+, PyTorch should detect your GPU out of the box. If you're getting weird HIP errors:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # for Vega/RDNA cards
export ROCM_PATH=/opt/rocm
```

The Radeon VII runs Whisper inference at roughly 3x realtime on the `base` model, which honestly surprised me. Fine-tuning the `small` model took about 4 hours on a 500-hour dataset. Not bad for a card that came out in 2019!

## Project Structure

```
├── src/
│   ├── whisper_finetune.py   # Fine-tune Whisper with ROCm workarounds
│   ├── stt_pipeline.py       # Speech-to-text inference
│   ├── tts_engine.py         # Coqui TTS wrapper
│   ├── diarizer.py           # Speaker diarization
│   └── audio_augment.py      # Audio augmentation pipeline
├── scripts/
│   ├── transcribe.py         # CLI transcription tool
│   └── eval_wer.py           # WER evaluation
├── configs/
│   └── whisper_finetune.yaml
└── experiments/
    └── notes.md              # My messy experiment notes
```

## Requirements

- Python 3.9+
- ROCm 5.x+ (tested on 5.5, 5.7)
- AMD GPU with at least 8GB VRAM (Radeon VII, RX 6800+, etc.)

## Hardware

Primary test target: AMD Radeon VII (ROCm 5.7). Some tests also run on CPU for baseline comparison.

## License

MIT — do whatever you want with it. Just don't blame me if your GPU catches fire (it won't).

---

*Built with a Radeon VII and too much coffee ☕*
