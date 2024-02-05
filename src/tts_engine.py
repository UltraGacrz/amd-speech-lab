"""
Text-to-Speech Engine (Coqui TTS)

Wrapper around Coqui TTS for easy text-to-speech synthesis.
Should work fine on ROCm since Coqui uses PyTorch under the hood.

Usage:
    from src.tts_engine import TTSEngine
    engine = TTSEngine()
    engine.synthesize("Hello world!", output_path="output.wav")
"""

import logging
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-speech using Coqui TTS on AMD GPUs."""

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        device: Optional[str] = None,
        vocoder: Optional[str] = None,
    ):
        """
        Initialize TTS engine.

        Args:
            model_name: Coqui TTS model path. Default is Tacotron2 with HiFi-GAN.
            device: 'cuda' or 'cpu'. Auto-detects if None.
            vocoder: Override vocoder model (optional).
        """
        # Lazy import — TTS is heavy and takes a sec
        from TTS.api import TTS

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading TTS model: {model_name}")
        logger.info(f"Device: {self.device}")

        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        self.tts = TTS(model_name=model_name, gpu=(self.device == "cuda"))
        self.model_name = model_name

        logger.info("TTS engine loaded ✓")

    def synthesize(
        self,
        text: str,
        output_path: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Synthesize text to audio.

        Args:
            text: Text to speak.
            output_path: Where to save the .wav file.
            speaker: Speaker ID for multi-speaker models.
            language: Language for multi-lingual models.

        Returns:
            Path to the output file.
        """
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        kwargs = {"text": text, "file_path": output_path}
        if speaker:
            kwargs["speaker"] = speaker
        if language:
            kwargs["language"] = language

        logger.info(f"Synthesizing: '{text[:50]}...' → {output_path}")
        self.tts.to_file(**kwargs)
        logger.info(f"Saved: {output_path}")

        return output_path

    def synthesize_to_array(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize text and return numpy array.
        Useful for real-time / streaming applications.
        """
        kwargs = {"text": text}
        if speaker:
            kwargs["speaker"] = speaker
        if language:
            kwargs["language"] = language

        wav = self.tts.tts(**kwargs)
        return np.array(wav, dtype=np.float32)

    def list_models(self) -> List[str]:
        """List available TTS models."""
        from TTS.utils.manage import ModelManager

        manager = ModelManager()
        return manager.list_models()

    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
    ) -> str:
        """
        Voice cloning (for models that support it).
        Provide a reference audio file of the target speaker.

        Note: Requires a model that supports voice cloning
        (e.g., YourTTS, XTTS).
        """
        logger.info(f"Cloning voice from: {reference_audio}")

        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            file_path=output_path,
        )

        return output_path


def main():
    """Quick test."""
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-speech synthesis")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", default="output.wav", help="Output path")
    parser.add_argument("--model", default="tts_models/en/ljspeech/tacotron2-DDC")
    parser.add_argument("--speaker", default=None, help="Speaker ID (multi-speaker)")
    args = parser.parse_args()

    engine = TTSEngine(model_name=args.model)
    engine.synthesize(args.text, args.output, speaker=args.speaker)
    print(f"Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
