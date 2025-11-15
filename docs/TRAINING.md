# Training Guide - Dysarthric Speech Recognition

---

## Table of Contents

1. [Training Configuration](#training-configuration)
2. [Training Process](#training-process)
3. [Monitoring Training](#monitoring-training)

---

### Training Strategy

**Multilingual Acoustic Transfer Learning:**
- Train on both English (TORGO) and Italian (EasyCall) datasets
- Use low learning rate (5e-6) to preserve pre-trained English knowledge
- Learn universal dysarthric acoustic patterns
- Force English output at inference time

---

## Training Configuration

### Basic Training Command

```bash
python src/step5_finetune.py --epochs 10 --batch-size 2 
```

### Parameters Explained

| Parameter | Default | Description | Recommended Values |
|-----------|---------|-------------|-------------------|
| `--epochs` | 10 | Number of training epochs | 10-20 |
| `--batch-size` | 2 | Samples per batch | 2-8 (CPU), 8-32 (GPU) |
| `--lr` | 5e-6 | Learning rate | 1e-6 to 1e-5 |

---

## Training Process

### Starting Training

```bash
# Activate environment
conda activate whisper_training

# Start training
python src/step5_finetune.py --epochs 10 --batch-size 2
```

### Expected Output

```
Epochs: 15
Batch: 4
LR: 5e-6

Loading dataset...
Loaded train: 32900 samples
Loaded val: 7050 samples
Loaded test: 7050 samples

Preparing dataset...
Preparing train: 100% [progress bar] 32900/32900 [10:30<00:00]
Preparing val: 100% [progress bar] 7050/7050 [02:15<00:00]

Loading model...
WhisperForConditionalGeneration loaded

TRAINING STARTED

Epoch 1/15:
Step 100/2057: loss=1.234, wer=45.2%
Step 200/2057: loss=1.123, wer=42.1%
...
```

### Training Timeline

**CPU (Intel i7, 16GB RAM):**
- ~40 minutes per epoch
- ~8 hours total for 10 epochs

**GPU (RTX 3060, 8GB VRAM):**
- ~6 minutes per epoch
- ~1.5 hours total for 15 epochs

---

## Resuming Training

If training is interrupted:

```bash
# Resume from last checkpoint
python src/step5_finetune.py --epochs 10 --batch-size 2 --resume
```

 Loads the last saved checkpoint.

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
