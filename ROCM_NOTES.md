# ROCm Notes — AMD Speech Lab

## Test Target

- **ROCm version:** 6.x (6.2 preferred)
- **PyTorch:** ROCm-enabled build
- **GPU strategy:** Single GPU inference, then multi-GPU for batch transcription
- **Primary card:** Radeon VII (16GB HBM2)

## Current Blockers

- Whisper's `flash_attention_2` kernel doesn't compile cleanly on ROCm — falling back to standard attention
- Coqui TTS vocoder (HiFi-GAN) has a custom CUDA kernel that needs ROCm porting; currently running via PyTorch fallback

## Planned Tests

| Test | Metric | Status |
|------|--------|--------|
| Whisper base inference | Realtime factor (RTF) | Pending |
| Whisper large-v3 inference | RTF, VRAM usage | Pending |
| fp16 Whisper encoder | Encoder throughput (ms/chunk) | Pending |
| bf16 Whisper encoder | Encoder throughput (ms/chunk) | Pending |
| TTS vocoder speed | Audio seconds generated per wall sec | Pending |
| Batch transcription | Throughput at batch 1/4/8/16 | Pending |
| Whisper encoder/decoder split | Per-component latency | Pending |

## Repo-Specific Notes

### Whisper Encoder/Decoder Split

The Whisper pipeline has two distinct GPU workload profiles:

- **Encoder:** Processes 30-second audio mel spectrograms. This is a convolution + transformer encoder stack. Highly parallelizable — should see strong GPU utilization on ROCm. Expected ~5-8ms per 30s chunk on RX 7900 XTX in fp16.
- **Decoder:** Autoregressive token generation. Each step requires a full forward pass but with tiny batch. This is latency-bound, not throughput-bound. Expect ~2-4ms per token.

The encoder is where GPU acceleration matters most for batch processing. The decoder bottleneck is sequential dependency, not compute.

### TTS Vocoder VRAM

- HiFi-GAN vocoder uses ~1.2GB VRAM at inference (fp32)
- With fp16, drops to ~0.7GB — significant for running alongside the acoustic model
- Full TTS pipeline (acoustic model + vocoder) peaks at ~4.5GB for single-speaker inference
- Batch synthesis (multiple texts) can push to 8-10GB depending on audio length

### Whisper Fine-tuning on ROCm

- Gradient checkpointing is essential for fine-tuning `large-v3` — without it, OOM on 16GB
- bf16 training is more stable than fp16 for Whisper — the cross-attention scores can overflow in fp16
- Data loading with `librosa` is CPU-bound — use `num_workers=4` minimum
- The CTC loss computation has no ROCm-specific issues

### Known ROCm Quirks for Speech

- `torchaudio` backend: use `soundfile` instead of `sox` — `sox` backend has segfault issues on ROCm systems
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` needed for Radeon VII (gfx906)
- Whisper's `torch.stft` works correctly on ROCm — no issues with mel spectrogram computation
- Memory fragmentation on long transcription runs — restart inference every ~100 files to avoid OOM creep
