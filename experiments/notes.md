# Experiment Notes

Messy notes from my experiments. Not meant to be pretty.

---

## 2024-01-15: Whisper base on Radeon VII

Finally got Whisper running on my Radeon VII with ROCm 5.5.

Key findings:
- `openai/whisper-base` runs at ~3x realtime
- `openai/whisper-small` runs at ~1.5x realtime
- `openai/whisper-medium` is about 0.7x (sub-realtime), needs more testing
- fp16 works perfectly on Vega VII

Had to set `HSA_OVERRIDE_GFX_VERSION=10.3.0` otherwise it crashes with some
cryptic HIP error. Classic ROCm stuff.

---

## 2024-01-22: Fine-tuning Whisper on LibriSpeech

Started fine-tuning whisper-base on LibriSpeech-100h.

Config:
- batch_size=8, grad_accum=2 (effective batch=16)
- lr=1e-5, warmup=500
- fp16 enabled
- gradient_checkpointing ON (otherwise OOM on 16GB)

Initial WER: 12.3% (base model) → after 5 epochs: 8.7%
After 10 epochs: 8.2% — diminishing returns after epoch 8.

Training time: ~4 hours on Radeon VII.

Note: torch.compile is BROKEN on ROCm. Just disable it. Save yourself the headache.

---

## 2024-02-03: Adding TTS with Coqui

Got Coqui TTS working with tacotron2-DDC. Sounds decent!
Also tried:
- VITS: faster and sounds more natural, but took longer to train
- XTTS v2: voice cloning is cool but needs a good reference clip

ROCm compatibility: no issues with Coqui TTS, it just uses standard PyTorch ops.

---

## 2024-02-10: Speaker diarization with pyannote

pyannote.audio 3.1 works on ROCm! The trick is:
1. Install pyannote.audio from pip
2. Accept the model terms on HuggingFace
3. Pass your HF token

Diarization error rate (DER) on my test meeting:
- 2 speakers: ~8% DER
- 4 speakers: ~15% DER (expected, harder problem)

Combining diarization with Whisper transcription = basically a poor man's
whisperX. Works well enough for my needs.

---

## 2024-02-18: Audio augmentation for training

Implemented augmentations:
- Gaussian noise: works great, biggest bang for buck
- Time stretch: useful but can introduce artifacts at extreme rates
- Pitch shift: +/- 3 semitones seems like the sweet spot
- Time masking: good for SpecAugment-style regularization
- Background noise mixing: need a good noise corpus

My noise corpus: ~200 clips of office/café/street noise, ~30s each.

Augmented my 100h LibriSpeech training data to effectively 300h.
WER dropped from 8.2% to 7.1%. Not bad!

---

## 2024-03-01: WER evaluation tooling

Built eval_wer.py. Key metrics:
- WER (Word Error Rate)
- CER (Character Error Rate)
- Substitutions, deletions, insertions breakdown

Using jiwer for computation. Clean evaluation pipeline.

Benchmark results (whisper-base fine-tuned, LibriSpeech test-clean):
- WER: 7.1%
- CER: 2.3%

For comparison, official Whisper base reports ~6.7% on test-clean.
My slightly worse result is likely because I fine-tuned on train-100
instead of the full 960h. Still pretty good!

---

## TODO / Next Steps

- [ ] Try whisper-large-v3 on the VII (might need gradient checkpointing++ and lower batch)
- [ ] Implement streaming STT for real-time applications
- [ ] Try XTTS v2 for better voice cloning
- [ ] Add beam search with LM for better accuracy
- [ ] Test on RX 7900 XTX (RDNA3) — should be even faster
- [ ] Build a simple web UI with Gradio
- [ ] Quantize the model for faster inference (INT8 on ROCm?)
