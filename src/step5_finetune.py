#!/usr/bin/env python3

# this is the core training script. fine-tunes OpenAI's pretrained Whisper model on the
# combined dysarthric speech dataset using multilingual acoustic transfer learning.

# torch: PyTorch deep learning framework, transformers: Hugging Face library with Whisper model
# evaluate: Metrics (WER - Word Error Rate), dataclasses: For data collator class
# logging: Training logs, gc: Garbage collection (memory management)


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
import gc
import sys

# we needed to fix windows console encoding issues; prevents crashes if displaying italian characters or progress bars
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# saves to training.log for review, displays in real-time.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# the datacollator pads batches so that they are the same duration. allows for finetuning
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
        raise FileNotFoundError("Cannot find combined_torgo_easycall directory!")
   
    logging.info(f"Using base: {base}")
   
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
       
        # Limit samples for memory
        original_count = len(data)
        if split == "train" and len(data) > 15000:
            logging.warning(f"[WARNING] Limiting {split} to 15000 samples (was {original_count})")
            data = data[:15000]
       
        valid_samples = []
        skipped = 0
       
        for item in data:
            audio_path = Path(item['audio'])
            if not audio_path.is_absolute():
                audio_path = base / audio_path
            if audio_path.exists():
                item['audio'] = str(audio_path.absolute())
                valid_samples.append(item)
            else:
                skipped += 1
       
        if not valid_samples:
            raise ValueError(f"No valid samples in {split}!")
       
        if skipped > 0:
            logging.warning(f"[WARNING] Skipped {skipped} missing files in {split}")
       
        dataset = Dataset.from_dict({
            "audio": [item['audio'] for item in valid_samples],
            "transcript": [item['transcript'] for item in valid_samples]
        })
       
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        datasets[split] = dataset
       
        logging.info(f"[OK] Loaded {split}: {len(dataset)} samples")
   
    return DatasetDict(datasets)

def prepare_dataset(dataset_dict, processor):
    def prepare_batch(batch):
        # encode input. audio features like sampling rate
        audio = batch["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        # decoder (text tokens, transcript)
        labels = processor.tokenizer(batch["transcript"]).input_ids
        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch
   
    prepared = {}
    for split, dataset in dataset_dict.items():
        logging.info(f"Preparing {split}")
       
        prepared[split] = dataset.map(
            prepare_batch,
            remove_columns=dataset.column_names,
            desc=f"Preparing {split}",
            batch_size=10,
            writer_batch_size=10,
            num_proc=1,
            keep_in_memory=False,
            load_from_cache_file=True
        )
       
        gc.collect()
        logging.info(f"[OK] Prepared {split}: {len(prepared[split])} samples")
   
    return DatasetDict(prepared)

def finetune(epochs=10, batch_size=2, lr=5e-6):
    print(f"\n Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   15k max samples")
    print("="*70 + "\n")
   
    # Find base directory
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
   
    #  1: Load dataset
    print("Loading dataset")
    dataset_dict = load_splits()
   
    #  2: Load processor
    print("\nLoading processor")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-base",
        task="transcribe"
    )
    logging.info("[OK] Processor loaded")
   
    #  3: Prepare dataset
    print("\nPreparing dataset")
    dataset_dict = prepare_dataset(dataset_dict, processor)
   
    #  4: Load model
    print("\nLoading model")
    # Load pre-trained Whisper as a starting point. This is the seq2seq model (encodes audio --> decodes into text)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    logging.info("[OK] Model loaded")
   
    #  5: Data collator
    print("\nCreating data collator")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    logging.info("Data collator ready")
   
    #  6: Metric
    print("\nLoading WER metric")
    metric = evaluate.load("wer")
    logging.info("[OK] Metric loaded")
   
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
   
    # STEP 7: Training args
    print("\nConfiguring training")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_steps=50,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=[],
        remove_unused_columns=False,
        weight_decay=0.01,
        dataloader_num_workers=0,
        dataloader_pin_memory=False
    )
    logging.info("[OK] Training args set")
   
    # STEP 8: Create trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("val"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )
    logging.info("[OK] Trainer ready")
   
    print("\n Starting training")
    print("\n" + "="*70)
    print(" TRAINING IN PROGRESS")
    print("="*70)
    print(f" Duration: ~{epochs * 30} minutes on CPU")
    print(f" Samples: {len(dataset_dict['train'])} train, {len(dataset_dict['val'])} val")
    print(f" Log: training.log")
    print("="*70 + "\n")
   
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted")
        logging.warning("Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        logging.error(f"Failed: {e}")
        raise
   
    # Save
    print("\n" + "="*70)
    print(" SAVING MODEL")
    print("="*70)
   
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
   
    info = {
        "datasets": "TORGO + EasyCall",
        "strategy": "Multilingual Acoustic Transfer",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_samples": len(dataset_dict["train"]),
        "val_samples": len(dataset_dict["val"]),
        "usage": "model.transcribe(audio, language='en')"
    }
   
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
   
    print(f"\n[OK] Saved: {output_dir}")
    print(" TRAINING COMPLETE!")

# THIS IS THE MAIN ENTRY POINT
if __name__ == "__main__":
    import argparse
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
   
    args = parser.parse_args()
   
    # NOW CALL THE FUNCTION WITH ARGUMENTS
    finetune(args.epochs, args.batch_size, args.lr)
