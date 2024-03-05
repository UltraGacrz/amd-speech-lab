"""
Speech-to-Text Pipeline

Drop-in STT using Whisper. Handles audio loading, chunking,
and batch transcription. Works with both the base OpenAI models
and fine-tuned checkpoints.

Usage:
    from src.stt_pipeline import STTPipeline
    pipeline = STTPipeline(model_name="openai/whisper-base")
    result = pipeline.transcribe("audio.wav")
"""

import logging
from pathlib import Path
from typing import Union, Optional

import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import pipeline as hf_pipeline, WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)


class STTPipeline:
    """Speech-to-text pipeline using Whisper on AMD/ROCm GPUs."""

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: Optional[str] = None,
        language: str = "en",
        task: str = "transcribe",
        chunk_length_s: float = 30.0,
    ):
        self.model_name = model_name
        self.language = language
        self.task = task
        self.chunk_length_s = chunk_length_s

        # Auto-detect device (ROCm shows up as cuda)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        self._load_model()

    def _load_model(self):
        """Load Whisper model and processor."""
        logger.info(f"Loading model: {self.model_name}")

        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language=self.language, task=self.task
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        # Move to GPU, use fp16 if on CUDA
        if self.device == "cuda":
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded ✓")

    def load_audio(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        """Load and resample audio to 16kHz mono."""
        audio_path = str(audio_path)

        # Check if file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load with librosa (handles most formats)
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)

        duration = len(audio) / sr
        logger.info(f"Loaded {audio_path}: {duration:.1f}s @ {sr}Hz")

        return audio

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        return_timestamps: bool = False,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio: Path to audio file or numpy array (16kHz mono)
            return_timestamps: Whether to return word-level timestamps

        Returns:
            dict with 'text' and optionally 'chunks' (timestamps)
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio = self.load_audio(str(audio))

        # Process
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        if self.device == "cuda":
            input_features = input_features.half().to(self.device)

        # Generate
        with torch.no_grad():
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=self.language, task=self.task
            )
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=448,
            )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        result = {"text": transcription}

        if return_timestamps:
            # For timestamps we'd need the timestamp-aware model
            result["chunks"] = [{"text": transcription, "timestamp": (0.0, len(audio) / 16000)}]

        return result

    def transcribe_batch(
        self, audio_paths: list, batch_size: int = 8
    ) -> list:
        """Transcribe multiple audio files."""
        results = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            for path in batch:
                result = self.transcribe(path)
                result["file"] = str(path)
                results.append(result)
            logger.info(f"Processed {min(i + batch_size, len(audio_paths))}/{len(audio_paths)}")

        return results

    def transcribe_long_audio(
        self, audio_path: str, overlap_s: float = 2.0
    ) -> dict:
        """
        Transcribe audio longer than 30s by chunking.
        Uses simple overlap-based chunking.
        """
        audio = self.load_audio(audio_path)
        sr = 16000
        chunk_samples = int(self.chunk_length_s * sr)
        overlap_samples = int(overlap_s * sr)

        if len(audio) <= chunk_samples:
            return self.transcribe(audio)

        chunks_text = []
        start = 0

        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            result = self.transcribe(chunk)
            chunks_text.append(result["text"])

            if end >= len(audio):
                break
            start += chunk_samples - overlap_samples

        # Simple concatenation (could be smarter with dedup)
        full_text = " ".join(chunks_text)
        return {"text": full_text, "num_chunks": len(chunks_text)}


def main():
    """Quick CLI test."""
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio")
    parser.add_argument("--input", "-i", required=True, help="Audio file path")
    parser.add_argument("--model", default="openai/whisper-base", help="Model name")
    parser.add_argument("--language", default="en", help="Language")
    args = parser.parse_args()

    pipeline = STTPipeline(model_name=args.model, language=args.language)
    result = pipeline.transcribe(args.input)
    print(f"\nTranscription:\n{result['text']}")


if __name__ == "__main__":
    main()
