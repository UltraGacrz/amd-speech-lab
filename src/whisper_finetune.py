"""
Whisper Fine-tuning on AMD GPUs (ROCm)

Handles all the fun quirks of running HuggingFace trainer on ROCm.
Main gotcha: torch.compile is basically broken on ROCm as of 2024,
and flash-attention support is spotty. We work around both.

Usage:
    python src/whisper_finetune.py --config configs/whisper_finetune.yaml
"""

import os
import argparse
import logging
from pathlib import Path

import yaml
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_rocm():
    """Verify ROCm setup and print GPU info."""
    if not torch.cuda.is_available():
        logger.warning("CUDA/ROCm not available! Training will be very slow on CPU.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ROCm-specific env vars
    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        logger.warning(
            "HSA_OVERRIDE_GFX_VERSION not set. "
            "If you get HIP errors, try: export HSA_OVERRIDE_GFX_VERSION=10.3.0"
        )

    return True


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(batch, feature_extractor, tokenizer):
    """Process audio and text for Whisper training."""
    # Load and resample audio
    audio = batch["audio"]
    
    # Compute log-mel spectrogram
    input_features = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Tokenize text
    batch["input_features"] = input_features
    batch["labels"] = tokenizer(batch["text"]).input_ids

    return batch


def compute_metrics(pred, tokenizer, metric):
    """Compute WER during training."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in labels (padding token)
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def apply_rocm_workarounds(config: dict):
    """
    ROCm doesn't always play nice with PyTorch features.
    This function applies known workarounds.
    """
    rocm_cfg = config.get("rocm", {})

    # torch.compile is broken on ROCm (as of PyTorch 2.x)
    if not rocm_cfg.get("compile_model", False):
        torch._dynamo.config.suppress_errors = True
        logger.info("torch.compile disabled (ROCm workaround)")

    # Set memory-efficient attention if flash-attn not available
    if rocm_cfg.get("mem_efficient_attention", True):
        os.environ["PYTORCH_ENABLE_MEM_EFFICIENT_SDPA"] = "1"
        logger.info("Using memory-efficient attention (ROCm)")

    # Disable flash attention if specified
    if not rocm_cfg.get("use_flash_attention", False):
        os.environ["PYTORCH_ENABLE_FLASH_ATTENTION"] = "0"
        os.environ["PYTORCH_ENABLE_SDPA"] = "0"
        logger.info("Flash attention disabled (ROCm workaround)")

    # Gradient checkpointing to save VRAM
    logger.info("ROCm workarounds applied ✓")


def main(config_path: str):
    config = load_config(config_path)
    
    # ROCm checks
    check_rocm()
    apply_rocm_workarounds(config)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Load processor, feature extractor, tokenizer
    logger.info(f"Loading model: {model_cfg['name']}")
    processor = WhisperProcessor.from_pretrained(
        model_cfg["name"], language=model_cfg["language"], task=model_cfg["task"]
    )
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["name"])

    # ROCm: force fp16 if configured
    if config.get("rocm", {}).get("fp16", True) and torch.cuda.is_available():
        model = model.half()
        logger.info("Model converted to fp16")

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=model_cfg["language"], task=model_cfg["task"]
    )
    model.config.suppress_tokens = []

    # Load dataset
    logger.info(f"Loading dataset: {data_cfg['dataset_name']}")
    dataset = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config", "clean"),
        trust_remote_code=True,
    )

    # Cast audio to correct sampling rate
    sampling_rate = config.get("preprocessing", {}).get("sampling_rate", 16000)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Process datasets
    num_proc = config.get("preprocessing", {}).get("num_proc", 4)
    train_dataset = dataset[data_cfg["train_split"]].map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=dataset[data_cfg["train_split"]].column_names,
        num_proc=num_proc,
    )
    eval_dataset = dataset[data_cfg["eval_split"]].map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=dataset[data_cfg["eval_split"]].column_names,
        num_proc=num_proc,
    )

    # WER metric
    wer_metric = evaluate.load("wer")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg.get("fp16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        logging_steps=train_cfg["logging_steps"],
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=config["evaluation"]["max_new_tokens"],
        report_to=["tensorboard"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        # ROCm sometimes has issues with pin_memory
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, wer_metric),
        tokenizer=processor.feature_extractor,
    )

    # Train!
    logger.info("Starting training...")
    trainer.train()

    # Save
    trainer.save_model(train_cfg["output_dir"])
    processor.save_pretrained(train_cfg["output_dir"])
    logger.info(f"Model saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on AMD GPUs")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/whisper_finetune.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
