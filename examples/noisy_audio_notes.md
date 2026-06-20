# Noisy Audio Handling Notes

## Types of noise

### Background music
- Whisper handles this surprisingly well
- Music with lyrics can confuse the model
- Instrumental background is usually fine

### Street noise
- Traffic, honking, wind
- Whisper struggles with sudden loud noises
- Noise reduction helps significantly

### Multiple speakers
- Whisper transcribes all speakers into one text
- No speaker diarization (who said what)
- Cross-talk causes errors

### Echo/reverb
- Room echo in recordings
- Whisper handles mild echo well
- Strong echo causes word duplication

## Pre-processing pipeline

```python
import librosa
import noisereduce as nr

def preprocess_audio(audio_path, target_sr=16000):
    # Load and resample
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Noise reduction (first 0.5s as noise profile)
    noise_sample = audio[:int(0.5 * sr)]
    cleaned = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=sr)
    
    # Normalize volume
    cleaned = cleaned / np.max(np.abs(cleaned))
    
    return cleaned, sr
```

## What works

1. Noise reduction with noisereduce library (simple, effective)
2. Volume normalization (helps with quiet recordings)
3. Resampling to 16kHz (Whisper's expected sample rate)

## What doesn't work

1. Aggressive noise reduction: removes speech along with noise
2. Echo cancellation: too complex, often makes things worse
3. Speed adjustment: changing playback speed hurts transcription
