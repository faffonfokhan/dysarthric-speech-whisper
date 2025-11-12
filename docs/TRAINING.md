# Training Guide - Dysarthric Speech Recognition

**Author:** Naba Khan  
**Date:** 2025-11-12  
**Institution:** CityUHK GEF2024

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Configuration](#training-configuration)
4. [Training Process](#training-process)
5. [Monitoring Training](#monitoring-training)
6. [Model Evaluation](#model-evaluation)

---

## Overview

This guide covers the  training process for fine-tuning Whisper on dysarthric speech datasets.

### Training Strategy

**Multilingual Acoustic Transfer Learning:**
- Train on both English (TORGO) and Italian (EasyCall) datasets
- Use low learning rate (5e-6) to preserve pre-trained English knowledge
- Learn universal dysarthric acoustic patterns
- Force English output at inference time

---

## Dataset Preparation

### Step 1: Organize Project Structure

```bash
python src/step1_organize_structure.py
```

**Creates:**
```
combined_torgo_easycall/
├── torgo/
│   └── organized/
│       ├── audio/
│       └── transcripts/
├── easycall/
│   └── organized/
│       ├── audio/
│       └── transcripts/
└── combined/
    ├── processed/
    └── splits/
```

### Step 2: Organize TORGO Dataset

```bash
python src/step2_organize_torgo.py
```

**Input:** Raw TORGO files from downloaded archive  
**Output:** Organized audio files and transcripts with standardized naming

**Naming Convention:**
```
torgo_dys_[speaker]_[session]_[utterance].wav
torgo_ctl_[speaker]_[session]_[utterance].wav
```

### Step 3: Organize EasyCall Dataset

```bash
python src/step3_organize_easycall.py
```

**Input:** EasyCall dataset directory  
**Output:** Organized files matching project structure

### Step 4: Combine Datasets

```bash
python src/step4_combine_datasets.py
```

**Process:**
1. Validates audio quality (duration, amplitude, format)
2. Resamples all audio to 16kHz mono
3. Normalizes amplitude
4. Creates train/val/test splits (70/15/15)
5. Generates JSON metadata files

**Output:**
```
combined/
├── processed/          # ~47,000 WAV files at 16kHz
└── splits/
    ├── train.json      # 70% of data
    ├── val.json        # 15% of data
    └── test.json       # 15% of data
```

---

## Training Configuration

### Basic Training Command

```bash
python src/step5_finetune.py --epochs 15 --batch-size 4 --lr 5e-6
```

### Parameters Explained

| Parameter | Default | Description | Recommended Values |
|-----------|---------|-------------|-------------------|
| `--epochs` | 15 | Number of training epochs | 10-20 |
| `--batch-size` | 4 | Samples per batch | 2-8 (CPU), 8-32 (GPU) |
| `--lr` | 5e-6 | Learning rate | 1e-6 to 1e-5 |

---

## Training Process

### Starting Training

```bash
# Activate environment
conda activate whisper_training

# Start training
python src/step5_finetune.py --epochs 15 --batch-size 4
```

### Expected Output

```
============================================================
FINE-TUNING WHISPER
User: faffonfokhan
Date: 2025-11-12 08:50:54 UTC
============================================================

Epochs: 15
Batch: 4
LR: 5e-6
============================================================

Loading dataset...
✅ Loaded train: 32900 samples
✅ Loaded val: 7050 samples
✅ Loaded test: 7050 samples

Preparing dataset...
Preparing train: 100%|████████████| 32900/32900 [10:30<00:00]
Preparing val: 100%|██████████████| 7050/7050 [02:15<00:00]

Loading model...
✅ WhisperForConditionalGeneration loaded

TRAINING STARTED

Epoch 1/15:
Step 100/2057: loss=1.234, wer=45.2%
Step 200/2057: loss=1.123, wer=42.1%
...
```

### Training Timeline

**CPU (Intel i7, 16GB RAM):**
- ~40 minutes per epoch
- ~10 hours total for 15 epochs

**GPU (RTX 3060, 8GB VRAM):**
- ~6 minutes per epoch
- ~1.5 hours total for 15 epochs

---

## Model Evaluation

### Testing on Held-out Test Set

After training completes:

```bash
python src/step7_test.py
```

**Output:**
```
Loading model...
✅ Model loaded

Testing on 5 samples:

Sample 1 (torgo):
  Reference: call mom
  Prediction: call mom
  Match: ✅

Sample 2 (easycall):
  Reference: scorri verso alto
  Prediction: scroll up
  Match: ✅ (cross-lingual success!)

...

Overall Results:
- Test WER: 24.3%
- Test CER: 12.1%
- Base Whisper WER: 52.7%
- Improvement: +28.4%
```

---

## Resuming Training

If training is interrupted:

```bash
# Resume from last checkpoint
python src/step5_finetune.py --epochs 15 --batch-size 4 --resume
```

Automatically loads the last saved checkpoint.

---

## Next Steps

After successful training:

1. [Convert Model](USAGE.md#model-conversion) to faster-whisper format
2. [Deploy Model](USAGE.md#deployment) for production use
3. [Integrate into Application](USAGE.md#integration)

---

**Last Updated:** 2025-11-12  
**Author:** Naba Khan  
**Institution:** CityUHK GEF2024
