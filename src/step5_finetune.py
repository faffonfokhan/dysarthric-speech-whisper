#!/usr/bin/env python3
"""
Fine-tune Whisper -  with absolute paths

"""

import json
from pathlib import Path
import torch
from datasets import Dataset, Audio, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
   
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
       
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
       
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
       
        batch["labels"] = labels
        return batch

def load_splits():
    """Load dataset splits with ABSOLUTE PATHS."""
   
    # Find base directory (try multiple locations)
    possible_bases = [
        Path("combined_torgo_easycall"),
        Path.cwd() / "combined_torgo_easycall",
        Path.home() / "combined_torgo_easycall",
    ]
   
    base = None
    for path in possible_bases:
        if path.exists():
            base = path.absolute()
            break
   
    if base is None:
        raise FileNotFoundError("Cannot find combined_torgo_easycall directory!")
   
    logging.info(f"Using base directory: {base}")
   
    splits_dir = base / "combined" / "splits"
   
    datasets = {}
   
    for split in ["train", "val", "test"]:
        split_file = splits_dir / f"{split}.json"
        if not split_file.exists():
            if split == "train":
                raise FileNotFoundError(f"Missing: {split_file}")
            continue
       
        with open(split_file, encoding='utf-8') as f:
            data = json.load(f)
       
        # FIX PATHS TO ABSOLUTE
        valid_samples = []
        for item in data:
            # Convert to absolute path
            audio_path = Path(item['audio'])
           
            # If relative, make absolute
            if not audio_path.is_absolute():
                audio_path = base / audio_path
           
            # Check if file exists
            if audio_path.exists():
                item['audio'] = str(audio_path.absolute())
                valid_samples.append(item)
            else:
                logging.warning(f"Missing file: {audio_path}")
       
        if not valid_samples:
            raise ValueError(f"No valid samples in {split}!")
       
        dataset = Dataset.from_dict({
            "audio": [item['audio'] for item in valid_samples],
            "transcript": [item['transcript'] for item in valid_samples]
        })
       
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        datasets[split] = dataset
       
        logging.info(f"Loaded {split}: {len(dataset)} samples")
   
    return DatasetDict(datasets)

def prepare_dataset(dataset_dict, processor):
    """Prepare dataset for training."""
    def prepare_batch(batch):
        audio = batch["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        labels = processor.tokenizer(batch["transcript"]).input_ids
        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch
   
    prepared = {}
    for split, dataset in dataset_dict.items():
        prepared[split] = dataset.map(
            prepare_batch,
            remove_columns=dataset.column_names,
            desc=f"Preparing {split}"
        )
    return DatasetDict(prepared)

def finetune(epochs=15, batch_size=4, lr=5e-6):
    """Fine-tune Whisper."""
    print("\n" + "="*60)
    print("FINE-TUNING WHISPER")
    print("User: faffonfokhan")
    print("Date: 2025-11-12 08:13:10 UTC")
    print("="*60)
    print(f"\nEpochs: {epochs}")
    print(f"Batch: {batch_size}")
    print(f"LR: {lr}")
    print("="*60 + "\n")
   
    # Find output directory
    possible_bases = [
        Path("combined_torgo_easycall"),
        Path.cwd() / "combined_torgo_easycall",
    ]
   
    base = None
    for path in possible_bases:
        if path.exists():
            base = path.absolute()
            break
   
    if base is None:
        base = Path("combined_torgo_easycall")
        base.mkdir(exist_ok=True)
   
    output_dir = base / "model" / "whisper_combined"
    output_dir.mkdir(parents=True, exist_ok=True)
   
    logging.info(f"Output: {output_dir}")
   
    # Load data
    dataset_dict = load_splits()
   
    # Load processor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-base",
        task="transcribe"
    )
   
    # Prepare
    dataset_dict = prepare_dataset(dataset_dict, processor)
   
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
   
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
   
    # Metric
    metric = evaluate.load("wer")
   
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
   
    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_steps=100,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        weight_decay=0.01
    )
   
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("val"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )
   
    print("TRAINING STARTED\n")
   
    # Train
    trainer.train()
   
    # Save
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
   
    print("\n✅ TRAINING COMPLETE!")
    print(f"Model: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    args = parser.parse_args()
   
    finetune(args.epochs, args.batch_size, args.lr)
