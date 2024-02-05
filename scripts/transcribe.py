#!/usr/bin/env python3
"""
Transcription CLI Tool

Quick and dirty transcription from the command line.
Supports single files, directories, and batch processing.

Usage:
    python scripts/transcribe.py --input audio.wav
    python scripts/transcribe.py --input ./audio_files/ --output results.json
    python scripts/transcribe.py --input audio.flac --model openai/whisper-small --language fr
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stt_pipeline import STTPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_audio_files(path: Path) -> list:
    """Find all audio files in a directory."""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}
    if path.is_file():
        return [path]

    files = []
    for ext in extensions:
        files.extend(path.glob(f"*{ext}"))
        files.extend(path.glob(f"**/*{ext}"))

    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper on AMD GPUs"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Audio file or directory",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    parser.add_argument(
        "--model", "-m",
        default="openai/whisper-base",
        help="Whisper model name or path",
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code",
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Handle long audio (>30s) with chunking",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for multiple files",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    # Find files
    audio_files = find_audio_files(input_path)
    if not audio_files:
        logger.error(f"No audio files found in: {args.input}")
        sys.exit(1)

    logger.info(f"Found {len(audio_files)} audio file(s)")

    # Init pipeline
    pipeline = STTPipeline(
        model_name=args.model,
        device=args.device,
        language=args.language,
    )

    # Transcribe
    results = []
    total_start = time.time()

    for i, audio_file in enumerate(audio_files):
        file_start = time.time()

        if args.long:
            result = pipeline.transcribe_long_audio(str(audio_file))
        else:
            result = pipeline.transcribe(str(audio_file))

        elapsed = time.time() - file_start
        result["file"] = str(audio_file)
        result["processing_time_s"] = round(elapsed, 2)
        results.append(result)

        logger.info(
            f"[{i+1}/{len(audio_files)}] {audio_file.name} "
            f"({elapsed:.1f}s): {result['text'][:80]}..."
        )

    total_elapsed = time.time() - total_start
    logger.info(f"Done! Total time: {total_elapsed:.1f}s")

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        for r in results:
            print(f"\n📁 {r['file']}")
            print(f"📝 {r['text']}")
            print(f"⏱️  {r['processing_time_s']}s")
        print()


if __name__ == "__main__":
    main()
