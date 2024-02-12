"""
Audio Augmentation Pipeline

Random audio augmentations for training data augmentation.
Uses audiomentations under the hood but provides a cleaner API.

Supported augmentations:
- Gaussian noise injection
- Time stretching (without pitch change)
- Pitch shifting
- Volume adjustment
- Random cropping
- Room reverb simulation
- Background noise mixing
- High/low pass filtering
- Time masking (SpecAugment-style)

Usage:
    from src.audio_augment import AudioAugmentor
    augmentor = AudioAugmentor(sr=16000)
    augmented = augmentor.augment(audio_array)
"""

import logging
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """Audio augmentation pipeline for speech data."""

    def __init__(
        self,
        sr: int = 16000,
        noise_dir: Optional[str] = None,
        p: float = 0.5,
    ):
        """
        Args:
            sr: Sample rate.
            noise_dir: Directory of noise files for mixing (optional).
            p: Probability of applying each augmentation.
        """
        self.sr = sr
        self.noise_dir = noise_dir
        self.p = p
        self._noise_files = []

        if noise_dir and Path(noise_dir).exists():
            self._noise_files = list(Path(noise_dir).glob("*.wav"))
            logger.info(f"Found {len(self._noise_files)} noise files")

    def augment(
        self,
        audio: np.ndarray,
        augmentations: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Apply random augmentations to audio.

        Args:
            audio: Input audio array.
            augmentations: List of augmentation names to apply. If None, applies all.

        Returns:
            Augmented audio array.
        """
        if augmentations is None:
            augmentations = [
                "gaussian_noise",
                "time_stretch",
                "pitch_shift",
                "volume",
                "time_mask",
                "high_low_pass",
            ]

        result = audio.copy()

        for aug_name in augmentations:
            if np.random.random() < self.p:
                result = getattr(self, f"_{aug_name}")(result)

        return result

    def augment_batch(
        self,
        audio_list: List[np.ndarray],
        n_augmented: int = 1,
    ) -> List[np.ndarray]:
        """Generate multiple augmented versions of each audio."""
        results = []
        for audio in audio_list:
            for _ in range(n_augmented):
                results.append(self.augment(audio))
        return results

    def _gaussian_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise."""
        noise_level = np.random.uniform(0.001, 0.015)
        noise = np.random.randn(len(audio)).astype(np.float32) * noise_level
        return audio + noise

    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Time stretch without changing pitch."""
        rate = np.random.uniform(0.85, 1.15)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return self._match_length(stretched, len(audio))

    def _pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Shift pitch in semitones."""
        n_steps = np.random.uniform(-3.0, 3.0)
        shifted = librosa.effects.pitch_shift(
            audio, sr=self.sr, n_steps=n_steps
        )
        return shifted

    def _volume(self, audio: np.ndarray) -> np.ndarray:
        """Random volume adjustment."""
        gain_db = np.random.uniform(-6.0, 6.0)
        gain = 10.0 ** (gain_db / 20.0)
        return audio * gain

    def _time_mask(self, audio: np.ndarray) -> np.ndarray:
        """Mask a random time segment (SpecAugment-style)."""
        max_mask_len = int(len(audio) * 0.1)  # up to 10% of audio
        if max_mask_len < 1:
            return audio

        mask_len = np.random.randint(1, max_mask_len + 1)
        start = np.random.randint(0, max(1, len(audio) - mask_len))

        result = audio.copy()
        result[start : start + mask_len] = 0.0
        return result

    def _high_low_pass(self, audio: np.ndarray) -> np.ndarray:
        """Apply random high-pass or low-pass filter."""
        filter_type = np.random.choice(["high", "low"])

        if filter_type == "high":
            cutoff = np.random.uniform(80, 300)
            filtered = self._butterworth_filter(audio, cutoff, "high")
        else:
            cutoff = np.random.uniform(3000, 8000)
            filtered = self._butterworth_filter(audio, cutoff, "low")

        return filtered

    def _butterworth_filter(
        self, audio: np.ndarray, cutoff: float, filter_type: str
    ) -> np.ndarray:
        """Simple Butterworth-style filter using librosa."""
        from scipy.signal import butter, sosfilt

        nyq = self.sr / 2
        cutoff_norm = min(cutoff / nyq, 0.99)

        if filter_type == "high":
            btype = "highpass"
        else:
            btype = "lowpass"

        sos = butter(4, cutoff_norm, btype=btype, output="sos")
        filtered = sosfilt(sos, audio).astype(np.float32)
        return filtered

    def _add_noise_from_file(self, audio: np.ndarray) -> np.ndarray:
        """Mix with a random noise file."""
        if not self._noise_files:
            return audio

        noise_path = np.random.choice(self._noise_files)
        noise, _ = librosa.load(str(noise_path), sr=self.sr, mono=True)

        # Match lengths
        if len(noise) < len(audio):
            # Repeat noise
            repeats = (len(audio) // len(noise)) + 1
            noise = np.tile(noise, repeats)
        noise = noise[: len(audio)]

        # Random SNR
        snr_db = np.random.uniform(5, 20)
        snr = 10.0 ** (snr_db / 20.0)

        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power > 0:
            scale = np.sqrt(audio_power / (snr ** 2 * noise_power))
            return audio + noise * scale
        return audio

    def _reverb(self, audio: np.ndarray) -> np.ndarray:
        """Simple reverb simulation using exponential decay."""
        decay = np.random.uniform(0.3, 0.8)
        reverb_len = int(self.sr * np.random.uniform(0.05, 0.2))

        impulse = np.zeros(reverb_len, dtype=np.float32)
        impulse[0] = 1.0
        for i in range(1, reverb_len):
            impulse[i] = decay ** i * np.random.uniform(-0.5, 0.5)

        from scipy.signal import fftconvolve
        reverbed = fftconvolve(audio, impulse, mode="full")[: len(audio)]
        return reverbed.astype(np.float32)

    def _match_length(self, audio: np.ndarray, target_len: int) -> np.ndarray:
        """Match audio length to target by trimming or padding."""
        if len(audio) > target_len:
            return audio[:target_len]
        elif len(audio) < target_len:
            padding = np.zeros(target_len - len(audio), dtype=audio.dtype)
            return np.concatenate([audio, padding])
        return audio

    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> np.ndarray:
        """Load audio file."""
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio

    @staticmethod
    def save_audio(audio: np.ndarray, path: str, sr: int = 16000):
        """Save audio file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, sr)


def main():
    """Quick demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Audio augmentation")
    parser.add_argument("--input", "-i", required=True, help="Input audio")
    parser.add_argument("--output", "-o", default="augmented.wav", help="Output path")
    parser.add_argument("--n", type=int, default=5, help="Num augmented versions")
    args = parser.parse_args()

    augmentor = AudioAugmentor()
    audio = AudioAugmentor.load_audio(args.input)

    stem = Path(args.output).stem
    ext = Path(args.output).suffix

    for i in range(args.n):
        augmented = augmentor.augment(audio)
        out_path = str(Path(args.output).parent / f"{stem}_{i}{ext}")
        AudioAugmentor.save_audio(augmented, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
