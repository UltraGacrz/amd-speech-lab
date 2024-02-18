"""
Speaker Diarization

Who's talking? Uses pyannote.audio for speaker diarization.
Note: pyannote requires a HuggingFace token for the pretrained models.

Usage:
    from src.diarizer import Diarizer
    diarizer = Diarizer()
    segments = diarizer.diarize("meeting.wav")
    for seg in segments:
        print(f"{seg['speaker']}: {seg['start']:.1f}s - {seg['end']:.1f}s")
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict

import torch
import numpy as np

logger = logging.getLogger(__name__)


class Diarizer:
    """Speaker diarization using pyannote.audio."""

    def __init__(
        self,
        pipeline_name: str = "pyannote/speaker-diarization-3.1",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Initialize diarization pipeline.

        Args:
            pipeline_name: pyannote pipeline identifier.
            device: 'cuda' or 'cpu'.
            hf_token: HuggingFace token (needed for pyannote models).
            num_speakers: Fixed number of speakers (if known).
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
        """
        from pyannote.audio import Pipeline

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        logger.info(f"Loading diarization pipeline: {pipeline_name}")
        logger.info(f"Device: {self.device}")

        self.pipeline = Pipeline.from_pretrained(
            pipeline_name,
            use_auth_token=hf_token,
        )

        if self.device == "cuda":
            self.pipeline = self.pipeline.to(torch.device(self.device))
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        logger.info("Diarization pipeline loaded ✓")

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: Override number of speakers for this file.

        Returns:
            List of segments: [{'speaker': str, 'start': float, 'end': float}, ...]
        """
        audio_path = str(audio_path)
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        logger.info(f"Diarizing: {audio_path}")

        # Build kwargs
        kwargs = {}
        ns = num_speakers or self.num_speakers
        if ns:
            kwargs["num_speakers"] = ns
        elif self.min_speakers:
            kwargs["min_speakers"] = self.min_speakers
        elif self.max_speakers:
            kwargs["max_speakers"] = self.max_speakers

        # Run pipeline
        diarization = self.pipeline(audio_path, **kwargs)

        # Convert to list of dicts
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3),
            })

        speakers = set(s["speaker"] for s in segments)
        logger.info(
            f"Found {len(speakers)} speakers, {len(segments)} segments"
        )

        return segments

    def diarize_with_transcript(
        self,
        audio_path: str,
        transcript: str,
        num_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """
        Combine diarization with a transcript.
        Very basic alignment — just assigns transcript chunks to speakers.

        For better alignment, use whisper + diarization together
        (like in whisperX).
        """
        segments = self.diarize(audio_path, num_speakers)

        if not segments:
            return [{"speaker": "unknown", "start": 0, "end": 0, "text": transcript}]

        # Naive: split transcript roughly proportionally by segment duration
        total_duration = sum(s["duration"] for s in segments)
        words = transcript.split()

        word_idx = 0
        for seg in segments:
            proportion = seg["duration"] / total_duration
            n_words = max(1, int(len(words) * proportion))
            seg_words = words[word_idx : word_idx + n_words]
            seg["text"] = " ".join(seg_words)
            word_idx += n_words

        # Remaining words go to last segment
        if word_idx < len(words):
            segments[-1]["text"] += " " + " ".join(words[word_idx:])

        return segments

    def get_speaker_stats(self, segments: List[Dict]) -> Dict:
        """Get speaking time statistics per speaker."""
        stats = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in stats:
                stats[spk] = {"total_time": 0, "num_segments": 0}
            stats[spk]["total_time"] += seg["duration"]
            stats[spk]["num_segments"] += 1

        # Round
        for spk in stats:
            stats[spk]["total_time"] = round(stats[spk]["total_time"], 2)

        return stats


def main():
    """CLI test."""
    import argparse

    parser = argparse.ArgumentParser(description="Speaker diarization")
    parser.add_argument("--input", "-i", required=True, help="Audio file")
    parser.add_argument("--speakers", "-n", type=int, default=None, help="Num speakers")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    args = parser.parse_args()

    diarizer = Diarizer(hf_token=args.hf_token)
    segments = diarizer.diarize(args.input, num_speakers=args.speakers)

    print("\nSpeaker Diarization Results:")
    print("-" * 50)
    for seg in segments:
        print(f"  {seg['speaker']:>10} | {seg['start']:>7.1f}s → {seg['end']:>7.1f}s | {seg['duration']:.1f}s")

    stats = diarizer.get_speaker_stats(segments)
    print(f"\nSpeaker Stats:")
    for spk, st in stats.items():
        print(f"  {spk}: {st['total_time']:.1f}s ({st['num_segments']} segments)")


if __name__ == "__main__":
    main()
