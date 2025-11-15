# Setup Guide - Dysarthric Speech Recognition

**Author:** Naba Khan  
**Date:** 2025-11-12  
**Institution:** CityUHK GEF2024

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Access](#dataset-access)
4. [Installation](#installation)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 16GB | 32GB |
| **Storage** | 50GB free | 100GB free |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | None (CPU works) | NVIDIA GPU with 8GB+ VRAM |

### Software Requirements

- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** 3.10 or higher
- **Conda:** Anaconda or Miniconda
- **Git:** For repository management

---

## Environment Setup

### Step 1: Install Conda

**Windows:**
```bash
# Download from: https://docs.conda.io/en/latest/miniconda.html
# Run installer and follow prompts
```

**Linux/macOS:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Clone Repository

```bash
git clone https://github.com/faffonfokhan/dysarthric-speech-whisper.git
cd dysarthric-speech-whisper
```

### Step 3: Create Python Environment

```bash
# Create environment with Python 3.10
conda create -n whisper_training python=3.10 -y

# Activate environment
conda activate whisper_training
```

### Step 4: Install Dependencies

```bash
# Install in correct order to avoid conflicts
pip install numpy==1.24.3
pip install pyarrow==12.0.1
pip install torch==2.0.1 torchaudio==2.0.2
pip install transformers==4.35.0 datasets==2.14.0
pip install evaluate==0.4.0 jiwer==3.0.0 accelerate==0.21.0
pip install librosa==0.10.0 soundfile==0.12.0
```

**OR** install from requirements file:

```bash
pip install -r requirements.txt
```

---

## Installation

### Verify Installation

After installing dependencies, verify everything works:

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from transformers import WhisperProcessor; print('Transformers: OK')"
python -c "from datasets import Dataset, Audio; print('Datasets: OK')"
python -c "import librosa; print('Librosa:', librosa.__version__)"
```

**Expected output:**
```
NumPy: 1.24.3
PyTorch: 2.0.1+cpu
Transformers: OK
Datasets: OK
Librosa: 0.10.0
```

---

## Start

After setup is complete:

```bash
# 1. Organize project structure
python src/step1_organize_structure.py

# 2. Organize TORGO dataset
python src/step2_organize_torgo.py

# 3. Organize EasyCall dataset
python src/step3_organize_easycall.py

# 4. Combine datasets
python src/step4_combine_datasets.py

# 5. Start training (8-24 hours on CPU)
python src/step5_finetune.py --epochs 10 --batch-size 2

# 6. Convert to faster-whisper
python src/step6_convert.py

```
---

## Next Steps

After successful setup, see:
- [Training Guide](TRAINING.md) - Detailed training instructions
- [Usage Guide](USAGE.md) - How to use trained model
- [Main README](../README.md) - Project overview
