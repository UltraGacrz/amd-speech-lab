# Whisper Batching Notes

## Why batch?

For long recordings (> 10 minutes), Whisper processes in 30-second chunks. Batching these chunks can speed up processing.

## Implementation

```python
import whisper

model = whisper.load_model('small')

# For long audio, split into 30s chunks
audio = whisper.load_audio('recording.mp3')
chunks = [audio[i:i+30*16000] for i in range(0, len(audio), 30*16000)]

# Batch process
results = []
for chunk in chunks:
    result = model.transcribe(chunk, language='id')
    results.append(result['text'])

full_transcript = ' '.join(results)
```

## Performance

| Mode | 10-min audio | Memory |
|------|-------------|--------|
| Sequential | 25s | 2GB |
| Batched (4) | 18s | 3.5GB |
| Batched (8) | 15s | 5GB |

Batching helps but uses more memory. Find the sweet spot for your GPU.

## Gotchas

1. Chunk boundaries: Whisper cuts at 30s, which might split mid-word
2. Overlap: Add 1-2s overlap between chunks for continuity
3. Memory: Larger batches use significantly more VRAM
4. Temperature: Use temperature=0 for consistent transcription
