# CPU Baseline Benchmarks — AMD Speech Lab

All benchmarks run on CPU without GPU acceleration.

## Environment

- **CPU:** AMD Ryzen 9 7950X (16C/32T)
- **RAM:** 64GB DDR5-6000
- **OS:** Ubuntu 22.04
- **Python:** 3.10
- **PyTorch:** 2.3.0 (CPU-only for baseline)

## Whisper Inference — Base Model

| Metric | Value |
|--------|-------|
| 30s audio transcription | 4.2 sec |
| Realtime factor (RTF) | 0.14 (7x faster than realtime) |
| Encoder latency (30s chunk) | 1,850 ms |
| Decoder latency (per token) | 18 ms |
| Peak memory (RSS) | 2.1 GB |
| Output: tokens/sec | 55 |

## Whisper Inference — Small Model

| Metric | Value |
|--------|-------|
| 30s audio transcription | 12.8 sec |
| Realtime factor (RTF) | 0.43 (2.3x realtime) |
| Encoder latency (30s chunk) | 5,200 ms |
| Decoder latency (per token) | 42 ms |
| Peak memory (RSS) | 3.4 GB |
| Output: tokens/sec | 24 |

## Whisper Inference — Large-v3 Model

| Metric | Value |
|--------|-------|
| 30s audio transcription | 58.3 sec |
| Realtime factor (RTF) | 1.94 (slower than realtime) |
| Encoder latency (30s chunk) | 28,400 ms |
| Decoder latency (per token) | 180 ms |
| Peak memory (RSS) | 8.7 GB |
| Output: tokens/sec | 5.5 |

## TTS — Coqui TTS (Tacotron2 + HiFi-GAN)

| Metric | Value |
|--------|-------|
| 10 sec audio synthesis | 22.5 sec |
| Peak memory (RSS) | 3.8 GB |
| Tokens processed/sec | 120 |
| Vocoder generation | 14.2 sec for 10s audio |

## Fine-tuning — Whisper Base (500hr dataset)

| Metric | Value |
|--------|-------|
| 1 epoch time | ~9.5 hours |
| Peak memory (RSS) | 6.2 GB |
| Batch size (max) | 8 |

---

**AMD GPU benchmark is pending — no access to ROCm hardware yet.**

Expected improvements on RX 7900 XTX (16GB HBM2):
- Whisper large-v3 encoder: ~15-25x speedup (transformer encoder is GPU-friendly)
- Decoder per-token: ~5-8x speedup (smaller gains due to autoregressive bottleneck)
- TTS vocoder: ~10-15x speedup (convolution-heavy, excellent GPU scaling)
- Overall: Whisper large-v3 should go from 0.5x realtime to 8-12x realtime
- Fine-tuning: ~6-10x speedup, enabling larger batch sizes and faster iteration
