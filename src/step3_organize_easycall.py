#!/usr/bin/env python3

from pathlib import Path
import shutil
from tqdm import tqdm

# Has the same overall aim as step2_organize_togo.py, but uses a database that has a preorganized structure,
# and handles transcripts easier, as this database was better organized. Code had lower complexity

def organize_easycall():
    # Find EasyCall
    easycall_source = Path("easycall_finetuning/2_organized")
   
    if not easycall_source.exists():
        print("EasyCall not found at: easycall_finetuning/2_organized/")
        print("\nTry these locations:")
        print("1. easycall_finetuning/2_organized/")
        print("2. Enter custom path")
       
        choice = input("\nChoice (1 or 2): ").strip()
       
        if choice == "2":
            custom = input("Enter EasyCall path: ").strip()
            easycall_source = Path(custom)
   
    if not easycall_source.exists():
        print(f"Not found: {easycall_source}")
        return False
   
    # Find files
    audio_src = easycall_source / "audio"
    trans_src = easycall_source / "transcripts"
   
    audio_files = list(audio_src.glob("*.wav"))
   
    if not audio_files:
        print("No audio files found!")
        return False
   
    print(f"Found {len(audio_files)} audio files")
    print("Organizing...\n")
   
    # Output
    base = Path("combined_torgo_easycall")
    audio_out = base / "easycall" / "organized" / "audio"
    trans_out = base / "easycall" / "organized" / "transcripts"
   
    stats = {"total": 0, "dysarthric": 0, "control": 0}
   
    # Copy files
    for audio_file in tqdm(audio_files, desc="Processing EasyCall"):
        # Copy audio
        dest_audio = audio_out / audio_file.name
        shutil.copy(audio_file, dest_audio)
       
        # Copy transcript
        trans_file = trans_src / f"{audio_file.stem}.txt"
        if trans_file.exists():
            dest_trans = trans_out / f"{audio_file.stem}.txt"
            shutil.copy(trans_file, dest_trans)
       
        stats["total"] += 1
        if "dys" in audio_file.stem.lower():
            stats["dysarthric"] += 1
        else:
            stats["control"] += 1
   
    print(f"Total: {stats['total']}")
    print(f"Dysarthric: {stats['dysarthric']}")
    print(f"Control: {stats['control']}")
    
if __name__ == "__main__":
    organize_easycall()
