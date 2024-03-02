#!/usr/bin/env python3
"""
Word Error Rate (WER) Evaluation

Evaluate transcription quality against ground truth.
Uses the jiwer library for WER computation.

Usage:
    python scripts/eval_wer.py --predictions preds.json --references refs.txt
    python scripts/eval_wer.py --model openai/whisper-base --test-data test.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for WER computation."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text


def compute_wer(predictions: List[str], references: List[str]) -> dict:
    """
    Compute WER and related metrics.

    Returns dict with: wer, cer, substitutions, deletions, insertions
    """
    # Normalize
    pred_norm = [normalize_text(p) for p in predictions]
    ref_norm = [normalize_text(r) for r in references]

    # WER
    wer = jiwer.wer(ref_norm, pred_norm)

    # CER (Character Error Rate)
    cer = jiwer.cer(ref_norm, pred_norm)

    # Detailed metrics
    measures = jiwer.compute_measures(ref_norm, pred_norm)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
        "num_sentences": len(predictions),
    }


def load_predictions(path: str) -> Tuple[List[str], List[str]]:
    """Load predictions file. Supports JSON and text formats."""
    path = Path(path)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            # List of dicts with 'text' and optionally 'reference'
            predictions = [d["text"] for d in data]
            references = [d.get("reference", "") for d in data]
        elif isinstance(data, dict):
            predictions = data.get("predictions", [])
            references = data.get("references", [])
        else:
            raise ValueError(f"Unexpected JSON format in {path}")
    else:
        # Plain text: one prediction per line
        with open(path) as f:
            predictions = [line.strip() for line in f if line.strip()]
        references = []

    return predictions, references


def load_references(path: str) -> List[str]:
    """Load reference transcripts."""
    path = Path(path)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [d.get("text", d.get("reference", "")) for d in data]
        return data.get("references", [])
    else:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]


def evaluate_model_on_dataset(
    model_name: str,
    test_data_path: str,
    language: str = "en",
    device: str = None,
) -> dict:
    """
    Run a model on test data and compute WER.
    Useful for comparing models.
    """
    from src.stt_pipeline import STTPipeline

    # Load test data
    test_data_path = Path(test_data_path)
    if test_data_path.suffix == ".json":
        with open(test_data_path) as f:
            test_data = json.load(f)
    else:
        logger.error(f"Unsupported format: {test_data_path.suffix}")
        sys.exit(1)

    references = [d["text"] for d in test_data]
    audio_files = [d["audio"] for d in test_data]

    # Transcribe
    pipeline = STTPipeline(
        model_name=model_name,
        device=device,
        language=language,
    )

    predictions = []
    for i, audio_file in enumerate(audio_files):
        result = pipeline.transcribe(audio_file)
        predictions.append(result["text"])
        logger.info(f"[{i+1}/{len(audio_files)}] {audio_file}")

    # Compute WER
    metrics = compute_wer(predictions, references)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER")
    parser.add_argument(
        "--predictions", "-p",
        default=None,
        help="Predictions file (JSON or text)",
    )
    parser.add_argument(
        "--references", "-r",
        default=None,
        help="Reference transcripts file",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Run model evaluation (model name)",
    )
    parser.add_argument(
        "--test-data", "-t",
        default=None,
        help="Test dataset JSON (for --model mode)",
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save results to JSON",
    )

    args = parser.parse_args()

    if args.model and args.test_data:
        # Model evaluation mode
        logger.info(f"Evaluating model: {args.model}")
        metrics = evaluate_model_on_dataset(
            args.model, args.test_data, args.language
        )
    elif args.predictions and args.references:
        # File comparison mode
        predictions, _ = load_predictions(args.predictions)
        references = load_references(args.references)

        if len(predictions) != len(references):
            logger.warning(
                f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
            )
            # Truncate to shorter
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]

        metrics = compute_wer(predictions, references)
    else:
        parser.error("Provide either --predictions + --references, or --model + --test-data")
        return

    # Print results
    print("\n" + "=" * 50)
    print("📊 WER Evaluation Results")
    print("=" * 50)
    print(f"  WER:            {metrics['wer']:.2%}")
    print(f"  CER:            {metrics['cer']:.2%}")
    print(f"  Substitutions:  {metrics['substitutions']}")
    print(f"  Deletions:      {metrics['deletions']}")
    print(f"  Insertions:     {metrics['insertions']}")
    print(f"  Hits:           {metrics['hits']}")
    print(f"  Sentences:      {metrics['num_sentences']}")
    print("=" * 50)

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
