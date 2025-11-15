#!/usr/bin/env python3

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import threading
import queue
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BulletproofTTS:
    def __init__(self):
        self.available = False
        self.method = "None"
        self.lock = threading.Lock()
       
        try:
            import pythoncom
            import win32com.client
           
            pythoncom.CoInitialize()
            test = win32com.client.Dispatch("SAPI.SpVoice")
            test.Speak("Ready", 0)
            del test
            pythoncom.CoUninitialize()
           
            self.available = True
            self.method = "Windows SAPI"
            logging.info("Windows SAPI ready")
            return
           
        except Exception as e:
            logging.warning(f"Windows SAPI failed: {e}")
       
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
           
            self.gTTS = gTTS
            self.pygame = pygame
            self.available = True
            self.method = "Google TTS"
            logging.info("Google TTS ready")
           
        except Exception as e:
            logging.error(f"All TTS failed: {e}")
            self.available = False
   
    def speak(self, text):
        if not self.available or not text.strip():
            return False
       
        with self.lock:
            logging.info(f"Speaking ({self.method}): {text[:50]}...")
           
            if self.method == "Windows SAPI":
                return self._speak_windows(text)
            elif self.method == "Google TTS":
                return self._speak_google(text)
           
            return False
   
    def _speak_windows(self, text):
        try:
            import pythoncom
            import win32com.client
           
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = 0
            speaker.Volume = 100
            speaker.Speak(text, 0)
            del speaker
            pythoncom.CoUninitialize()
           
            logging.info("Spoke successfully")
            return True
           
        except Exception as e:
            logging.error(f"Windows TTS error: {e}")
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            return False
   
    def _speak_google(self, text):
        try:
            import tempfile
            import os
           
            tts = self.gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                temp = f.name
           
            tts.save(temp)
           
            self.pygame.mixer.music.load(temp)
            self.pygame.mixer.music.play()
           
            while self.pygame.mixer.music.get_busy():
                self.pygame.time.Clock().tick(10)
           
            os.unlink(temp)
            logging.info("Spoke successfully")
            return True
           
        except Exception as e:
            logging.error(f"Google TTS error: {e}")
            return False
   
    def stop(self):
        try:
            if self.method == "Google TTS" and hasattr(self, 'pygame'):
                self.pygame.mixer.music.stop()
        except:
            pass

class FineTunedASR:
   
    def __init__(self):
        from faster_whisper import WhisperModel
       
        # Find any available converted model
        base = Path("combined_torgo_easycall") / "model"
       
        possible_models = []
       
        # Look for any converted CT2 models
        if base.exists():
            for item in base.iterdir():
                if item.is_dir() and item.name.endswith('_ct2'):
                    if (item / "model.bin").exists():
                        possible_models.append(item)
       
        if not possible_models:
            raise FileNotFoundError(
                "No converted model found!\n\n"
                "Expected: combined_torgo_easycall/model/*_ct2/\n\n"
                "Convert a checkpoint:\n"
                "  python step6_convert_checkpoint.py --checkpoint checkpoint-1000\n\n"
                "Or convert final model:\n"
                "  python step6_convert_checkpoint.py"
            )
       
        # Sort by name (latest will be last)
        possible_models.sort(key=lambda x: x.name)
       
        # Use latest model
        model_path = possible_models[-1]
       
        logging.info(f"Found converted models:")
        for m in possible_models:
            logging.info(f"  - {m.name}")
       
        logging.info(f"Using: {model_path.name}")
       
        # Load model
        self.model = WhisperModel(
            str(model_path),
            device="cpu",
            compute_type="int8",
            num_workers=4
        )
       
        self.model_path = model_path
        self.model_name = model_path.name
       
        # Try to load metadata
        self.metadata = {}
        metadata_file = model_path / "conversion_info.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                self.metadata = json.load(f)
       
        logging.info(f"Model loaded: {self.model_name}")
   
    def transcribe(self, audio):
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True
        )
       
        transcription = " ".join([s.text.strip() for s in segments])
       
        logging.info(f"Transcription: '{transcription}'")
        logging.info(f"Language: {info.language} ({info.language_probability:.2%})")
       
        return transcription
   
    def get_model_info(self):
        """Get model information."""
        checkpoint = self.metadata.get('checkpoint', 'unknown')
       
        return {
            "name": self.model_name,
            "path": str(self.model_path),
            "checkpoint": checkpoint,
            "type": "Fine-tuned Whisper-base",
            "training": "TORGO + EasyCall",
            "optimization": "Dysarthric speech"
        }

class AudioProc:
    def __init__(self):
        self.sr = 16000
   
    def record(self, duration):
        import sounddevice as sd
        import numpy as np
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype=np.float32)
        sd.wait()
        return audio.flatten()
   
    def load(self, path):
        import librosa
        return librosa.load(path, sr=self.sr, mono=True)[0]

# gui
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Assistant")
        self.root.geometry("950x800")
       
        self.tts = None
        self.asr = None
        self.audio = None
       
        self.q = queue.Queue()
       
        self.build_ui()
        self.root.after(100, self.init_tts)
        threading.Thread(target=self.init_asr, daemon=True).start()
        self.root.after(50, self.update)
   
    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
       
        tk.Label(
            header,
            text="🎤 Dysarthric Speech Assistant",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        ).pack(pady=8)
       
        # Status
        self.status = tk.StringVar(value="Initializing...")
        tk.Label(self.root, textvariable=self.status, font=("Arial", 10)).pack(pady=5)
       
        # Model info
        self.model_info = tk.StringVar(value="Loading model...")
        info_frame = tk.Frame(self.root, bg="#ecf0f1", relief=tk.RIDGE, bd=2)
        info_frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(
            info_frame,
            textvariable=self.model_info,
            font=("Arial", 9),
            bg="#ecf0f1",
            fg="#34495e"
        ).pack(pady=5)
       
        # Main buttons
        btn1 = tk.Frame(self.root)
        btn1.pack(pady=15)
       
        self.rec_btn = tk.Button(
            btn1,
            text="🎙️ Record (10s)",
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            width=16,
            height=2,
            command=self.record,
            state=tk.DISABLED
        )
        self.rec_btn.pack(side=tk.LEFT, padx=10)
       
        self.file_btn = tk.Button(
            btn1,
            text="📁 Load File",
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            width=16,
            height=2,
            command=self.load,
            state=tk.DISABLED
        )
        self.file_btn.pack(side=tk.LEFT, padx=10)
       
        # Text area
        tk.Label(self.root, text="Transcription:", font=("Arial", 11, "bold")).pack(anchor=tk.W, padx=20)
        self.text = scrolledtext.ScrolledText(self.root, font=("Arial", 13), height=8, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
       
        # Separator
        tk.Frame(self.root, height=2, bg="#ccc").pack(fill=tk.X, padx=20, pady=10)
       
        # TTS SECTION
        tk.Label(self.root, text="🔊 TEXT-TO-SPEECH", font=("Arial", 14, "bold"), fg="#ff6b6b").pack(pady=5)
       
        self.tts_status = tk.StringVar(value="Loading TTS...")
        tk.Label(self.root, textvariable=self.tts_status, font=("Arial", 9), fg="#666").pack()
       
        tts_buttons = tk.Frame(self.root)
        tts_buttons.pack(pady=10)
       
        self.speak_btn = tk.Button(tts_buttons, text="🔊 SPEAK TEXT", font=("Arial", 16, "bold"), bg="#ff6b6b", fg="white", width=16, height=2, command=self.speak, state=tk.DISABLED)
        self.speak_btn.grid(row=0, column=0, padx=8)
       
        self.test_btn = tk.Button(tts_buttons, text="Test TTS", font=("Arial", 14, "bold"), bg="#17a2b8", fg="white", width=12, height=2, command=self.test, state=tk.DISABLED)
        self.test_btn.grid(row=0, column=1, padx=8)
       
        self.stop_btn = tk.Button(tts_buttons, text="⏹️ Stop", font=("Arial", 14, "bold"), bg="#e67e22", fg="white", width=10, height=2, command=self.stop_tts, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=8)
       
        tk.Frame(self.root, height=2, bg="#ccc").pack(fill=tk.X, padx=20, pady=10)
       
        # Other buttons
        other = tk.Frame(self.root)
        other.pack(pady=10)
        tk.Button(other, text="🗑️ Clear", width=10, command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(other, text="📋 Copy", width=10, command=self.copy).pack(side=tk.LEFT, padx=5)
        tk.Button(other, text="💾 Save", width=10, command=self.save).pack(side=tk.LEFT, padx=5)
       
        # Footer
        footer = tk.Frame(self.root, bg="#ecf0f1")
        footer.pack(fill=tk.X, pady=5)
        tk.Label(footer, text="CityUHK GEF2024 | faffonfokhan | 2025-11-14", font=("Arial", 8), bg="#ecf0f1", fg="#7f8c8d").pack()
       
        logging.info("✅ UI built")
   
    def init_tts(self):
        logging.info("Initializing TTS...")
        self.tts_status.set("Loading TTS...")
       
        try:
            self.tts = BulletproofTTS()
           
            if self.tts.available:
                self.tts_status.set(f"✅ {self.tts.method} ready")
                self.speak_btn.config(state=tk.NORMAL)
                self.test_btn.config(state=tk.NORMAL)
                logging.info(f"✅ TTS ready: {self.tts.method}")
            else:
                self.tts_status.set("❌ TTS unavailable")
        except Exception as e:
            logging.error(f"TTS init error: {e}")
            self.tts_status.set(f"❌ TTS Error")
   
    def init_asr(self):
        self.q.put(("status", "Loading model... (30-60s)"))
       
        try:
            self.asr = FineTunedASR()
            self.audio = AudioProc()
           
            info = self.asr.get_model_info()
           
            self.q.put(("status", "✅ Model ready!"))
            self.q.put(("model_info",
                f"Model: {info['checkpoint']} | "
                f"Type: {info['type']} | "
                f"Datasets: {info['training']}"
            ))
            self.q.put(("enable_asr", None))
           
            logging.info("✅ ASR ready")
           
        except FileNotFoundError as e:
            logging.error(f"Model not found: {e}")
            self.q.put(("status", "❌ Model not found!"))
            self.q.put(("model_info", "Run: python step6_convert_checkpoint.py --checkpoint checkpoint-1000"))
            messagebox.showerror(
                "Model Not Found",
                f"{e}\n\nConvert a checkpoint first:\n"
                "python step6_convert_checkpoint.py --checkpoint checkpoint-1000"
            )
           
        except Exception as e:
            logging.error(f"ASR error: {e}")
            self.q.put(("status", f"❌ Error: {e}"))
            self.q.put(("model_info", "Failed to load - see console"))
   
    def test(self):
        if not self.tts or not self.tts.available:
            messagebox.showerror("Error", "TTS not available")
            return
       
        self.test_btn.config(state=tk.DISABLED)
        self.speak_btn.config(state=tk.DISABLED)
       
        def do():
            success = 0
            for i in range(1, 4):
                self.q.put(("status", f"🔊 Test {i}/3"))
                if self.tts.speak(f"This is test number {i}"):
                    success += 1
                else:
                    break
                time.sleep(0.5)
           
            if success == 3:
                self.q.put(("status", "✅ All tests passed!"))
                messagebox.showinfo("Success", "TTS works!")
            else:
                self.q.put(("status", f"❌ {success}/3 passed"))
           
            self.test_btn.config(state=tk.NORMAL)
            self.speak_btn.config(state=tk.NORMAL)
       
        threading.Thread(target=do, daemon=True).start()
   
    def speak(self):
        text = self.text.get(1.0, tk.END).strip()
        text = ' '.join([l for l in text.split('\n') if not l.startswith('[')])
       
        if not text:
            messagebox.showwarning("No Text", "Nothing to speak")
            return
       
        if not self.tts or not self.tts.available:
            messagebox.showerror("TTS Error", "TTS not available")
            return
       
        self.speak_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
       
        def do():
            self.q.put(("status", "🔊 Speaking..."))
            success = self.tts.speak(text)
            if success:
                self.q.put(("status", "✅ Done"))
            else:
                self.q.put(("status", "❌ Failed"))
           
            self.speak_btn.config(state=tk.NORMAL)
            self.test_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
       
        threading.Thread(target=do, daemon=True).start()
   
    def stop_tts(self):
        if self.tts:
            self.tts.stop()
        self.speak_btn.config(state=tk.NORMAL)
        self.test_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
   
    def record(self):
        self.rec_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
       
        def do():
            try:
                self.q.put(("status", "🎤 Recording 10s..."))
                audio = self.audio.record(10)
                self.q.put(("status", "Transcribing..."))
                text = self.asr.transcribe(audio)
                self.q.put(("append", f"\n[Recorded {time.strftime('%H:%M:%S')}]\n{text}\n"))
                self.q.put(("status", "✅ Done"))
            except Exception as e:
                self.q.put(("status", f"❌ {e}"))
            finally:
                self.rec_btn.config(state=tk.NORMAL)
                self.file_btn.config(state=tk.NORMAL)
       
        threading.Thread(target=do, daemon=True).start()
   
    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac")])
        if not path:
            return
       
        self.rec_btn.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)
       
        def do():
            try:
                self.q.put(("status", f"Loading {Path(path).name}..."))
                audio = self.audio.load(path)
                self.q.put(("status", "Transcribing..."))
                text = self.asr.transcribe(audio)
                self.q.put(("append", f"\n[File: {Path(path).name}]\n{text}\n"))
                self.q.put(("status", "✅ Done"))
            except Exception as e:
                self.q.put(("status", f"❌ {e}"))
            finally:
                self.rec_btn.config(state=tk.NORMAL)
                self.file_btn.config(state=tk.NORMAL)
       
        threading.Thread(target=do, daemon=True).start()
   
    def clear(self):
        self.text.delete(1.0, tk.END)
   
    def copy(self):
        text = self.text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Copied!")
   
    def save(self):
        text = self.text.get(1.0, tk.END).strip()
        if not text:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
   
    def update(self):
        try:
            while True:
                msg, data = self.q.get_nowait()
               
                if msg == "status":
                    self.status.set(data)
                elif msg == "model_info":
                    self.model_info.set(data)
                elif msg == "append":
                    self.text.insert(tk.END, data)
                    self.text.see(tk.END)
                elif msg == "enable_asr":
                    self.rec_btn.config(state=tk.NORMAL)
                    self.file_btn.config(state=tk.NORMAL)
        except queue.Empty:
            pass
       
        self.root.after(50, self.update)

def main():
    print(" Auto-detecting checkpoint model...")
    print("="*70 + "\n")
   
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
