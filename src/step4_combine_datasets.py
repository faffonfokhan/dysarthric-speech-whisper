#!/usr/bin/env python3
# This file merges the organized TORGO and EasyCall datasets.

from pathlib import Path
import json
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def validate_and_process(audio_file, trans_file, dataset_name, output_dir):

    # first, check whether transcript exists
    try:
        if not trans_file.exists():
            return None
       
        transcript = trans_file.read_text(encoding='utf-8').strip()
        if not transcript:
            return None
       
        # load audio, then validate duration. calculate duration by dividing audio samples by sample rate
        audio, sr = librosa.load(audio_file, sr=None)
        duration = len(audio) / sr

        # skip files shorter than 0.5 seconds (likely noise) and longer than 30 seconds (too long for speech recognition)
        if duration < 0.5 or duration > 30.0:
            return None

        # calculate root mean squared value to calculate loudness. if smaller than 0.001 (likely no noise, corrupted audio)
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.001:
            return None
       
        # standardize audio to whisper's standard rate, and a consistent volume
        audio = librosa.load(audio_file, sr=16000, mono=True)[0]
        audio = librosa.util.normalize(audio)
       
        # save processed audio
        output_name = f"{dataset_name}_{audio_file.stem}"
        output_file = output_dir / f"{output_name}.wav"
        sf.write(output_file, audio, 16000)
       
        # create metadata, which separates healthy from dysarthric udio and lanuages (english or italian)
        is_dys = "dys" in audio_file.stem.lower()
        language = "english" if dataset_name == "torgo" else "italian"

        # return dictionary with audio filename, text transcript, duration (rounded to 2 decimals), dysarthric flag (boolean)
        # source dataset and language
        return {
            "audio": str(output_file.name),
            "transcript": transcript,
            "duration": round(duration, 2),
            "dysarthric": is_dys,
            "dataset": dataset_name,
            "language": language
        }

    # handle error by skipping problematic files
    except Exception as e:
        return None

def combine_datasets():

    # define paths
    base = Path("combined_torgo_easycall")
   
    # Source directories
    torgo_audio = base / "torgo" / "organized" / "audio"
    torgo_trans = base / "torgo" / "organized" / "transcripts"
    easycall_audio = base / "easycall" / "organized" / "audio"
    easycall_trans = base / "easycall" / "organized" / "transcripts"
   
    # create output directories and an empty list for processed samples
    processed = base / "combined" / "processed"
    splits_dir = base / "combined" / "splits"
   
    processed.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
   
    all_samples = []
   
    # process the torgo dataset
    print("Processing TORGO")
    torgo_files = list(torgo_audio.glob("*.wav"))

    # find each audio file, locate transcript, validate length, and add to all_samples if satisfiable
    for audio_file in tqdm(torgo_files, desc="TORGO"):
        trans_file = torgo_trans / f"{audio_file.stem}.txt"
        sample = validate_and_process(audio_file, trans_file, "torgo", processed)
        if sample:
            all_samples.append(sample)

    # count and display valid samples
    torgo_count = len([s for s in all_samples if s["dataset"] == "torgo"])
    print(f"TORGO: {torgo_count} samples")
   
    # same process for easycall!!!
    print("\nProcessing EasyCall")
    easycall_files = list(easycall_audio.glob("*.wav"))
   
    for audio_file in tqdm(easycall_files, desc="EasyCall"):
        trans_file = easycall_trans / f"{audio_file.stem}.txt"
        sample = validate_and_process(audio_file, trans_file, "easycall", processed)
        if sample:
            all_samples.append(sample)
   
    easycall_count = len([s for s in all_samples if s["dataset"] == "easycall"])
    print(f"EasyCall: {easycall_count} samples")
   
    # display all samples
    print()
    print("="*60)
    print("COMBINED DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(all_samples)}")
    print(f"TORGO (English): {torgo_count}")
    print(f"EasyCall (Italian): {easycall_count}")
    print(f"Dysarthric: {sum(1 for s in all_samples if s['dysarthric'])}")
    print(f"Duration: {sum(s['duration'] for s in all_samples)/3600:.1f} hours")
   
    # Create splits
    print()
    print("="*60)
    print("CREATING SPLITS (70/15/15)")
    print("="*60)
   
    np.random.seed(42)
    np.random.shuffle(all_samples)

    # Split calculation: train: 70% (first portion), val: 15% (middle portion), test: 15% (remaining)
    
    n = len(all_samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
   
    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train+n_val],
        "test": all_samples[n_train+n_val:]
    }

    # save split files
    for split_name, samples in splits.items():
        # Update audio paths to be relative to base
        for sample in samples:
            sample["audio"] = f"combined/processed/{sample['audio']}"
       
        split_file = splits_dir / f"{split_name}.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"{split_name}: {len(samples)} samples")
   
    # Save metadata
    metadata = {
        "prepared_at": "2025-11-10 08:14:38 UTC",
        "user": "faffonfokhan",
        "total": len(all_samples),
        "torgo": torgo_count,
        "easycall": easycall_count,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"])
    }
   
    with open(base / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    combine_datasets()
