#!/usr/bin/env python3

from pathlib import Path
import json

def organize_structure():
    print("\n" + "="*60)
    print("ORGANIZING COMBINED DATASET STRUCTURE")
    print("User: faffonfokhan")
    print("Date: 2025-11-10 08:14:38 UTC")
    print("="*60 + "\n")
   
    base = Path("combined_torgo_easycall")
   
    dirs = {
        "TORGO Raw": base / "torgo" / "raw",
        "TORGO Organized": base / "torgo" / "organized" / "audio",
        "TORGO Transcripts": base / "torgo" / "organized" / "transcripts",
        "EasyCall Raw": base / "easycall" / "raw",
        "EasyCall Organized": base / "easycall" / "organized" / "audio",
        "EasyCall Transcripts": base / "easycall" / "organized" / "transcripts",
        "Combined Processed": base / "combined" / "processed",
        "Combined Splits": base / "combined" / "splits",
        "Model Output": base / "model",
        "Logs": base / "logs"
    }
   
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"{name:25s} → {path}/")
   
    config = {
        "user": "faffonfokhan",
        "created": "2025-11-10 08:14:38 UTC",
        "datasets": ["TORGO", "EasyCall"],
        "target_sample_rate": 16000
    }
   
    with open(base / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
   
    print("\Structure created!")
    print(f"\nProject: {base.absolute()}")
    print("\nNext: python step2_organize_torgo.py")

if __name__ == "__main__":
    organize_structure()
