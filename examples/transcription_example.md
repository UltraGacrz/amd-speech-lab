# Transcription Example — AMD Speech Lab

## Input

**Audio file:** `sample_meeting.wav`
- Duration: 2 minutes 34 seconds
- Format: WAV, 16kHz, mono
- Content: Two people discussing a project timeline in English
- Background noise: Light office ambient (air conditioning, keyboard clicks)

## Processing Command

```bash
python scripts/transcribe.py \
  --input sample_meeting.wav \
  --model openai/whisper-base \
  --language en \
  --output-format srt \
  --timestamps true
```

## Expected Output

```
[00:00:00.000 --> 00:00:04.200] Hey, so where are we on the Q3 deliverables?
[00:00:04.200 --> 00:00:08.500] We're mostly on track. The API integration is done,
[00:00:08.500 --> 00:00:12.100] but we're running behind on the frontend dashboard.
[00:00:12.100 --> 00:00:16.800] How far behind? Like a week or more?
[00:00:16.800 --> 00:00:21.300] Probably two weeks. The design team just finalized
[00:00:21.300 --> 00:00:25.900] the mockups last Friday, so we lost some time there.
[00:00:25.900 --> 00:00:30.200] Okay, can we pull someone from the backend team to help?
[00:00:30.200 --> 00:00:34.800] Yeah, Sarah's free after Thursday. She knows React.
...
```

## Performance Notes

| Metric | CPU (baseline) | Expected RX 7900 XTX |
|--------|----------------|----------------------|
| Transcription time | 18.2 sec | ~1.5 sec |
| Realtime factor | 8.5x | ~100x |
| Peak memory | 2.1 GB | 2.5 GB (VRAM) |
| WER (on LibriSpeech test) | 7.2% | 7.2% (same model) |

## TTS Example

**Input text:** "Welcome to the AMD Speech Lab. This is a text-to-speech demo running on ROCm."

```bash
python scripts/synthesize.py \
  --text "Welcome to the AMD Speech Lab. This is a text-to-speech demo running on ROCm." \
  --output welcome.wav \
  --voice default
```

**Expected output:** `welcome.wav` — 4.2 seconds of clear speech, ~110ms latency before audio starts playing.
