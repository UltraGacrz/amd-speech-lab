# amd-speech-lab

This repo is a small speech AI playground focused on transcription experiments, especially Indonesian audio, noisy recordings, and short voice notes. The first target is Whisper-style inference before adding more TTS experiments.

## Why I built this

I have a bunch of Indonesian voice notes and meeting recordings that I want to transcribe. Cloud speech APIs work but cost money per minute and send my audio to external servers. I wanted to see if local Whisper models are good enough for Indonesian.

## What's in here

- Whisper inference experiments on Indonesian audio
- Notes on noisy audio handling
- Batch transcription pipeline

## Current experiments

1. Whisper base/small/medium on Indonesian speech
2. Noise reduction preprocessing
3. Batch processing for long recordings

## Models tested

- Whisper base (74M)
- Whisper small (244M)
- Whisper medium (769M)

## Quick start

```bash
pip install -r requirements.txt
python speech_lab.py transcribe recording.mp3 --model small --language id
```

## Examples

- `examples/indonesian_transcription.md` -- transcription quality on Indonesian audio
- `examples/noisy_audio_notes.md` -- handling noisy recordings
