#!/usr/bin/env python3

from pathlib import Path
import shutil
from tqdm import tqdm
import zipfile
import tarfile

def extract_if_needed(torgo_path):
# finds all .zip and .tar files, 
    archives = list(torgo_path.glob("*.zip")) + list(torgo_path.glob("*.tar*"))
   
    if archives:
        print(f"Found {len(archives)} archive(s), extracting...")
        extract_dir = torgo_path / "extracted"
        extract_dir.mkdir(exist_ok=True)
      
        # Creates extracted subdirectory
        for archive in archives:
            print(f"  Extracting: {archive.name}")
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(extract_dir)
            else:
                with tarfile.open(archive) as tf:
                    tf.extractall(extract_dir)
     
        # returns extracted directory only if the data was extracted; otherwise it returns the original path
        return extract_dir
   
    return torgo_path

def organize_torgo():
    # Finds TORGO location
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
   
    # Find audio files and adds them to a list
    audio_files = []
    for ext in ["*.wav", "*.WAV"]:
        audio_files.extend(source_dir.rglob(ext))
   
    if not audio_files:
        print("No WAV files found!")
        print(f"Searched in: {source_dir}")
        return False
   
    print(f"\nFound {len(audio_files)} audio files")
    print("Organizing...\n")
   
    # setup all the output directories, splits into audio and transcripts
    base = Path("combined_torgo_easycall")
    audio_out = base / "torgo" / "organized" / "audio"
    trans_out = base / "torgo" / "organized" / "transcripts"

    # tracks file counts for healthy speakers and impaired speakers
    stats = {"total": 0, "dysarthric": 0, "control": 0}
   
    # processes audio files with a progress bar
    for audio_file in tqdm(audio_files, desc="Processing TORGO"):
        # identifies speaker type
        parts = audio_file.parts
        speaker_id = "unknown"
        is_dysarthric = False

        # detect dysarthric speakers from audio files.
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
       
        # standardise file names (dysarthric: torgo_dys_MC01_filename, control: torgo_ctl_FC01_filename)
        prefix = "torgo_dys" if is_dysarthric else "torgo_ctl"
        new_name = f"{prefix}_{speaker_id}_{audio_file.stem}"
       
        # copy audio files
        dest_audio = audio_out / f"{new_name}.wav"
        shutil.copy(audio_file, dest_audio)
       
        # Find transcript
        trans_found = False
        possible_trans = [
            audio_file.with_suffix('.txt'),
            audio_file.parent / "prompts" / f"{audio_file.stem}.txt",
            audio_file.parent.parent / "prompts" / f"{audio_file.stem}.txt"
        ]

        # search multiple locations for transcripts
        for trans_file in possible_trans:
            if trans_file.exists():
                text = trans_file.read_text(encoding='utf-8', errors='ignore').strip()
                dest_trans = trans_out / f"{new_name}.txt"
                dest_trans.write_text(text, encoding='utf-8')
                trans_found = True
                break

        # if tanscripts are not found, replace transcript with file name without underscores
        if not trans_found:
            # Generate from filename
            text = audio_file.stem.lower().replace('_', ' ')
            dest_trans = trans_out / f"{new_name}.txt"
            dest_trans.write_text(text, encoding='utf-8')

        # update statistics
        stats["total"] += 1
        if is_dysarthric:
            stats["dysarthric"] += 1
        else:
            stats["control"] += 1
            
    print(f"Total: {stats['total']}")
    print(f"Dysarthric: {stats['dysarthric']}")
    print(f"Control: {stats['control']}")


if __name__ == "__main__":
    organize_torgo()
