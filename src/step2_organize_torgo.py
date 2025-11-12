#!/usr/bin/env python3
"""
Organize TORGO dataset

"""

from pathlib import Path
import shutil
from tqdm import tqdm
import zipfile
import tarfile

def extract_if_needed(torgo_path):
    """Extract TORGO if it's an archive."""
    archives = list(torgo_path.glob("*.zip")) + list(torgo_path.glob("*.tar*"))
   
    if archives:
        print(f"Found {len(archives)} archive(s), extracting...")
        extract_dir = torgo_path / "extracted"
        extract_dir.mkdir(exist_ok=True)
       
        for archive in archives:
            print(f"  Extracting: {archive.name}")
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_dir)
            else:
                with tarfile.open(archive) as tf:
                    tf.extractall(extract_dir)
       
        return extract_dir
   
    return torgo_path

def organize_torgo():
    print("\n" + "="*60)
    print("ORGANIZING TORGO DATASET")
    print("User: faffonfokhan")
    print("Date: 2025-11-10 08:14:38 UTC")
    print("="*60 + "\n")
   
    # Find TORGO location
    print("Where is your TORGO dataset?")
    print("1. Enter path manually")
    print("2. Current directory")
   
    choice = input("\nChoice (1 or 2): ").strip()
   
    if choice == "1":
        torgo_input = input("Enter TORGO path: ").strip()
        torgo_path = Path(torgo_input)
    else:
        torgo_path = Path(".")
   
    if not torgo_path.exists():
        print(f"Path not found: {torgo_path}")
        return False
   
    # Extract if needed
    source_dir = extract_if_needed(torgo_path)
   
    # Find audio files
    audio_files = []
    for ext in ["*.wav", "*.WAV"]:
        audio_files.extend(source_dir.rglob(ext))
   
    if not audio_files:
        print("No WAV files found!")
        print(f"Searched in: {source_dir}")
        return False
   
    print(f"\nFound {len(audio_files)} audio files")
    print("Organizing...\n")
   
    # Output directories
    base = Path("combined_torgo_easycall")
    audio_out = base / "torgo" / "organized" / "audio"
    trans_out = base / "torgo" / "organized" / "transcripts"
   
    stats = {"total": 0, "dysarthric": 0, "control": 0}
   
    # Process files
    for audio_file in tqdm(audio_files, desc="Processing TORGO"):
        # Identify speaker
        parts = audio_file.parts
        speaker_id = "unknown"
        is_dysarthric = False
       
        for part in parts:
            part_upper = part.upper()
            # Dysarthric: FC*, MC*, F01, F03, M0*
            if part_upper.startswith(('FC', 'MC', 'F01', 'F03')) or (part_upper.startswith('M0') and not part_upper.startswith('M01')):
                speaker_id = part
                is_dysarthric = True
                break
            # Control: F03, F04, M01-M05
            elif part_upper.startswith(('F04', 'M01', 'M02', 'M03', 'M04', 'M05')):
                speaker_id = part
                is_dysarthric = False
                break
       
        # Create filename
        prefix = "torgo_dys" if is_dysarthric else "torgo_ctl"
        new_name = f"{prefix}_{speaker_id}_{audio_file.stem}"
       
        # Copy audio
        dest_audio = audio_out / f"{new_name}.wav"
        shutil.copy(audio_file, dest_audio)
       
        # Find transcript
        trans_found = False
        possible_trans = [
            audio_file.with_suffix('.txt'),
            audio_file.parent / "prompts" / f"{audio_file.stem}.txt",
            audio_file.parent.parent / "prompts" / f"{audio_file.stem}.txt"
        ]
       
        for trans_file in possible_trans:
            if trans_file.exists():
                text = trans_file.read_text(encoding='utf-8', errors='ignore').strip()
                dest_trans = trans_out / f"{new_name}.txt"
                dest_trans.write_text(text, encoding='utf-8')
                trans_found = True
                break
       
        if not trans_found:
            # Generate from filename
            text = audio_file.stem.lower().replace('_', ' ')
            dest_trans = trans_out / f"{new_name}.txt"
            dest_trans.write_text(text, encoding='utf-8')
       
        stats["total"] += 1
        if is_dysarthric:
            stats["dysarthric"] += 1
        else:
            stats["control"] += 1
   
    print()
    print("="*60)
    print("TORGO ORGANIZATION COMPLETE")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"Dysarthric: {stats['dysarthric']}")
    print(f"Control: {stats['control']}")
    print()
    print("Next: python step3_organize_easycall.py")

if __name__ == "__main__":
    organize_torgo()
