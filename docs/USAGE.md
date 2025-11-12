# Usage Guide - Dysarthric Speech Recognition

**Author:** Naba Khan  
**Date:** 2025-11-12  
**Institution:** CityUHK GEF2024

---

## Table of Contents

1. [Model Conversion](#model-conversion)
2. [Basic Usage](#basic-usage)
3. [Integration Examples](#integration-examples)

---

## Model Conversion

### Convert to Faster-Whisper Format

After training, convert model for faster inference:

```bash
python src/step6_convert.py
```

**Process:**
1. Loads PyTorch model
2. Converts to CTranslate2 format
3. Applies INT8 quantization (4x smaller, 3x faster)
4. Saves optimized model

**Output:**
```
combined_torgo_easycall/model/whisper_combined_ct2/
├── config.json
├── model.bin
├── vocabulary.txt
└── tokenizer.json
```

---

## Basic Usage

### Simple Transcription

```python
from faster_whisper import WhisperModel

# Load model
model = WhisperModel(
    "combined_torgo_easycall/model/whisper_combined_ct2",
    device="cpu",
    compute_type="int8"
)

# Transcribe audio
segments, info = model.transcribe(
    "path/to/audio.wav",
    language="en",  # Force English output
    beam_size=5
)

# Get transcription
transcription = " ".join([seg.text for seg in segments])
print(transcription)
```

### Batch Transcription

```python
import glob
from faster_whisper import WhisperModel

model = WhisperModel(
    "combined_torgo_easycall/model/whisper_combined_ct2",
    device="cpu",
    compute_type="int8"
)

# Process multiple files
audio_files = glob.glob("audio_folder/*.wav")

for audio_file in audio_files:
    segments, info = model.transcribe(audio_file, language="en")
    text = " ".join([seg.text for seg in segments])
    print(f"{audio_file}: {text}")
```
---

## Integration Examples

### With Streamlit (Web UI, our choice)

```python
import streamlit as st
from faster_whisper import WhisperModel
import tempfile

@st.cache_resource
def load_model():
    return WhisperModel(
        "combined_torgo_easycall/model/whisper_combined_ct2",
        device="cpu",
        compute_type="int8"
    )

st.title("Dysarthric Speech Transcriber")
st.write("Upload audio file for transcription")

model = load_model()

uploaded_file = st.file_uploader("Choose audio file", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            segments, info = model.transcribe(tmp_path, language="en")
            text = " ".join([seg.text for seg in segments])
        
        st.success("Transcription complete!")
        st.write("**Transcription:**")
        st.write(text)
        
        st.write("**Details:**")
        st.write(f"Duration: {info.duration:.2f}s")
        st.write(f"Language: {info.language}")
```

**Run:**
```bash
streamlit run app.py
```
---

**Last Updated:** 2025-11-12  
**Author:** Naba Khan  
**Institution:** CityUHK GEF2024
