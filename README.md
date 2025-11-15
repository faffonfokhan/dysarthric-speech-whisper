# Dysarthric Speech Recognition with Whisper Fine-tuning

**Authors:** Naba Khan, Owais Khan  
Fine-tuning OpenAI Whisper for dysarthric speech recognition using TORGO and EasyCall datasets.

---

## Overview

We aim tp improve speech recognition accuracy for people with dysarthria (motor speech disorder) by fine-tuning OpenAI's Whisper model on specialized datasets.

### Problem
- Standard speech recognition: 40-60% accuracy on dysarthric speech
- Our fine-tuned model: 70-85% accuracy

### Datasets Used
- **TORGO**: 8 dysarthric speakers (English)
- **EasyCall**: 26 dysarthric speakers (Italian)
- **Combined**: ~47,000 audio samples

---

## Results

| Model | Accuracy |
|-------|----------|
| Base Whisper | 40-60% |
| Fine-tuned (TORGO only) | 60-80% |
| Fine-tuned (EasyCall only) | 55-75% |
| **Fine-tuned (Combined,estimated)** | **70-85%** |

---

### Prerequisites
- Python 3.10+
- 16GB RAM minimum
- GPU recommended (CPU works but slower)

### Setup

# Create environment
conda create -n whisper_training python=3.10 -y
conda activate whisper_training

# Install dependencies
pip install -r requirements.txt
