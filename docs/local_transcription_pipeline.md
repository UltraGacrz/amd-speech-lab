# Local Transcription Pipeline

## Full pipeline

```
Raw audio file
    ↓ preprocess
Noise reduction + resampling
    ↓ transcribe
Whisper inference
    ↓ postprocess
Text cleanup + formatting
    ↓ output
SRT/TXT/JSON
```

## Step 1: Preprocessing

- Load audio with librosa
- Resample to 16kHz
- Apply noise reduction
- Normalize volume

## Step 2: Transcription

- Load Whisper model (small for speed, medium for quality)
- Transcribe with language='id'
- Use temperature=0 for consistency

## Step 3: Post-processing

- Remove filler words (optional)
- Fix common transcription errors
- Format as SRT with timestamps

## Step 4: Output

- SRT: for video subtitles
- TXT: plain text transcript
- JSON: with timestamps and confidence scores

## Costs

| Approach | 1 hour audio | Cost |
|----------|-------------|------|
| Google Speech API | 1 hour | ~$1.50 |
| Whisper local (small) | 1 hour | $0 (electricity only) |
| Whisper local (medium) | 1 hour | $0 (slower) |

Local is free after initial setup. Break-even vs cloud at ~10 hours of audio.
