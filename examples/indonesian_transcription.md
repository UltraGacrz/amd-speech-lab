# Indonesian Transcription Examples

## Test 1: Clear speech (podcast)

Input: 5-minute Indonesian podcast clip, clear studio audio.

| Model | WER | Speed | Notes |
|-------|-----|-------|-------|
| Whisper base | 12% | 8x realtime | Some misheard words |
| Whisper small | 7% | 3x realtime | Good quality |
| Whisper medium | 4% | 1x realtime | Near perfect |

WER = Word Error Rate. Lower is better.

## Test 2: Casual conversation

Input: 3-minute recording of friends talking casually, background noise.

| Model | WER | Notes |
|-------|-----|-------|
| Whisper base | 25% | Struggled with slang |
| Whisper small | 15% | Better but missed some words |
| Whisper medium | 10% | Good, occasional errors |

Casual Indonesian has a lot of slang and code-switching (mixing Indonesian/English). Models handle formal Indonesian much better.

## Test 3: Phone call recording

Input: 2-minute phone call, compressed audio, some static.

| Model | WER | Notes |
|-------|-----|-------|
| Whisper base | 30% | Very noisy |
| Whisper small | 20% | Better |
| Whisper medium | 14% | Usable |

Phone audio quality hurts transcription significantly.

## Common errors

1. Proper nouns: Names of people/places often misheard
2. Numbers: 'dua ratus' (two hundred) sometimes transcribed as 'dua ratus' or '200'
3. Code-switching: English words in Indonesian sentences cause errors
4. Fillers: 'eh', 'ya', 'gitu' sometimes dropped or misheard

## Recommendations

- For clear audio: Whisper small is good enough
- For noisy audio: Whisper medium is worth the extra compute
- Pre-processing: Noise reduction helps, but don't over-process
