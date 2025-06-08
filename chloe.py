import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinter import simpledialog
import threading
import subprocess
import shutil
import sys
import re
import gc
import os
import json
import time
import tempfile
import textwrap
import random
from datetime import datetime
import psutil
import requests
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, List, Optional, Any

PROFILE_COLL = "chloe_user_profile"
HIST_COLL = "chloe_convo_history"
PROFILE_JSON = "chloe_profile.json"
HIST_JSON = "chloe_convo.json"
CHROMADB_PATH = "./chromadb_data"

def ensure_model_ready(model_name="llama3.2"):
    setup_root = tk.Tk()
    setup_root.title("Setting up Chloe")
    setup_root.geometry("560x300")
    setup_root.configure(bg="#1a1f2e")

    label = tk.Label(
        setup_root,
        text=f"Preparing Chloe AI...\nDownloading model: {model_name}\n\n",
        bg="#1a1f2e", fg="#ffffff", font=("Segoe UI", 12)
    )
    label.pack(pady=(16, 0))

    text = tk.Text(setup_root, height=11, width=66, bg="#151a28", fg="#c7fffd", font=("Consolas", 10))
    text.pack(padx=18, pady=(0, 10))
    text.insert(tk.END, "Waiting for download to start...\n")
    text.config(state='disabled')

    status_label = tk.Label(setup_root, text="Initializing...", bg="#1a1f2e", fg="#b0bec5", font=("Segoe UI", 10))
    status_label.pack(pady=(0, 5))

    def do_setup():
        # Check if model already downloaded
        proc = subprocess.Popen(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        list_out = proc.communicate()[0].decode(errors='ignore')
        if model_name in list_out:
            status_label.config(text=f"{model_name} already downloaded")
            setup_root.after(1200, setup_root.destroy)
            return

        status_label.config(text=f"Downloading {model_name} ...")
        setup_root.update_idletasks()

        # Run ollama pull and display output
        proc = subprocess.Popen(['ollama', 'pull', model_name],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode(errors='ignore')
            setup_root.after(0, lambda d=decoded: append_text(d))
        proc.wait()

        # Loading phase (dummy, just for UI flow)
        status_label.config(text=f"Loading {model_name} ...")
        setup_root.update_idletasks()
        setup_root.after(1200, setup_root.destroy)

    def append_text(text_line):
        text.config(state='normal')
        text.insert(tk.END, text_line)
        text.see(tk.END)
        text.config(state='disabled')

    threading.Thread(target=do_setup, daemon=True).start()
    setup_root.mainloop()

def get_or_create_coll(name, embedding_model=None):
    try:
        os.makedirs(CHROMADB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMADB_PATH)
        try:
            return client.get_collection(name=name)
        except Exception:
            embed_fn = None
            if embedding_model == 'ollama':
                embed_fn = embedding_functions.OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",
                )
            return client.create_collection(name=name, embedding_function=embed_fn)
    except Exception:
        client = chromadb.Client()
        try:
            return client.get_collection(name=name)
        except Exception:
            return client.create_collection(name=name)

def save_profile(profile):
    try:
        c = get_or_create_coll(PROFILE_COLL, embedding_model='ollama')
        c.upsert(ids=["profile"], documents=[json.dumps(profile)])
    except Exception:
        pass
    with open(PROFILE_JSON, "w", encoding="utf-8") as f:
        json.dump(profile, f)

def load_profile():
    try:
        c = get_or_create_coll(PROFILE_COLL)
        res = c.get(ids=["profile"])
        if res and res.get('documents'):
            return json.loads(res['documents'][0])
    except Exception:
        pass
    if os.path.exists(PROFILE_JSON):
        try:
            with open(PROFILE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_history(history):
    try:
        c = get_or_create_coll(HIST_COLL, embedding_model='ollama')
        c.upsert(ids=["history"], documents=[json.dumps(history)])
    except Exception:
        pass
    with open(HIST_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f)

def load_history():
    try:
        c = get_or_create_coll(HIST_COLL)
        res = c.get(ids=["history"])
        if res and res.get('documents'):
            return json.loads(res['documents'][0])
    except Exception:
        pass
    if os.path.exists(HIST_JSON):
        try:
            with open(HIST_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def strip_action_tags(text):
    # Remove *text* or _text_ or [text] at start of lines, or in the body if they're action/tone cues
    text = re.sub(r'(\*|_)[^*_]{0,40}(\*|_)', '', text)
    # Remove [text] style cues too (if they're not bracketed links)
    text = re.sub(r'\[[a-zA-Z ,\-]+\]', '', text)
    # Remove common meta phrases at start of lines
    text = re.sub(r'^(?:\(.*?\)|\*.*?\*|_.*?_)', '', text, flags=re.MULTILINE)
    # Remove lines that are only meta cues
    text = re.sub(r'^\s*\*[^*]{0,80}\*\s*$', '', text, flags=re.MULTILINE)
    # Extra: If LLM outputs "Tone: gentle" or "Action: chuckles", strip those too
    text = re.sub(r'^\s*(Tone|Action)\s*:\s*\w+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def download_tiny_model_blocking():
    setup_root = tk.Tk()
    setup_root.title("Setting up Chloe")
    setup_root.geometry("400x180")
    setup_root.configure(bg="#1a1f2e")

    label = tk.Label(
        setup_root,
        text="First time setup:\nDownloading a small AI model (llama3.2)...\nPlease wait.",
        bg="#1a1f2e", fg="#ffffff", font=("Segoe UI", 12)
    )
    label.pack(pady=(30, 5))

    progress_var = tk.DoubleVar()
    progress = ttk.Progressbar(setup_root, orient='horizontal', length=300,
                               mode='determinate', variable=progress_var, maximum=100)
    progress.pack(pady=10)

    percent_label = tk.Label(setup_root, text="0%", bg="#1a1f2e", fg="#ffffff", font=("Segoe UI", 10))
    percent_label.pack()

    def do_download():
        # Spawn ollama pull and parse percent
        proc = subprocess.Popen(['ollama', 'pull', 'llama3.2'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        last_percent = 0
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode(errors='ignore')
            # Try to find percent in the output line
            m = re.search(r'\((\d+)%\)', decoded)
            if m:
                percent = int(m.group(1))
                progress_var.set(percent)
                percent_label.config(text=f"{percent}%")
                setup_root.update_idletasks()
                last_percent = percent
            elif "done" in decoded.lower():
                progress_var.set(100)
                percent_label.config(text="100%")
                setup_root.update_idletasks()
        proc.wait()
        setup_root.after(0, setup_root.destroy)

    threading.Thread(target=do_download, daemon=True).start()
    setup_root.mainloop()

# ----------- Pretty CLI Helpers for Ollama status ------------
def pretty_print(msg, color="cyan", bold=True):
    COLORS = {
        "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
        "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m",
        "gray": "\033[90m", "white": "\033[97m", "reset": "\033[0m"
    }
    prefix = COLORS.get(color, "")
    suffix = COLORS["reset"]
    style = "\033[1m" if bold else ""
    print(f"{style}{prefix}{msg}{suffix}")

def print_ollama_help():
    pretty_print("\n🚨 OLLAMA NOT FOUND 🚨\n", "red")
    print("To use Chloe with full features (LLM jokes, smart conversation, etc):")
    pretty_print("1. Download and install Ollama:", "cyan", False)
    print("   https://ollama.com/download")
    pretty_print("2. Run Ollama in the background:", "cyan", False)
    pretty_print("   ollama serve")
    pretty_print("3. Try running Chloe again!", "yellow", False)
    pretty_print("If you need help: https://github.com/ollama/ollama/blob/main/docs/index.md\n", "magenta")
    pretty_print("EXITING...", "yellow")

def check_ollama_full_status():
    # Returns (is_ollama_installed, is_ollama_running)
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, timeout=5)
        installed = True
    except Exception:
        installed = False
    if not installed:
        return (False, False)
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=3)
        running = (r.status_code == 200)
    except Exception:
        running = False
    return (installed, running)

class ModelManager:
    MODELS = {
        "light": {
            "name": "Lightweight",
            "models": ["llama3.2", "tinyllama"],
            "ram_req": 8,
            "description": "Fast & lightweight"
        },
        "medium": {
            "name": "Balanced (Recommended)",
            "models": ["mannix/llama3.1-8b-lexi", "llama3"],
            "ram_req": 16,
            "description": "Balanced for most users"
        },
        "heavy": {
            "name": "Heavyweight",
            "models": ["phi4", "qwen2.5:14b"],
            "ram_req": 32,
            "description": "Requires serious hardware"
        }
    }

    CACHE_TTL = 30  # seconds

    def __init__(self):
        self.available_ram = psutil.virtual_memory().total / (1024**3)
        self._downloaded_models = None
        self._last_check = None
        self._ollama_installed = None

    def _refresh_cache(self):
        if self._last_check is None or (datetime.now() - self._last_check).total_seconds() > self.CACHE_TTL:
            self._downloaded_models = None

    def get_downloaded_models(self):
        self._refresh_cache()
        if self._downloaded_models is not None:
            return self._downloaded_models
        models = []
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=10)
            if result.returncode == 0:
                try:
                    stdout = result.stdout.decode('utf-8', errors='replace')
                except Exception:
                    stdout = result.stdout.decode('latin-1', errors='replace')
                lines = stdout.strip().split('\n')
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        models.append(parts[0].strip())
            self._downloaded_models = models
            self._last_check = datetime.now()
        except Exception:
            self._downloaded_models = []
        return self._downloaded_models

    def is_model_downloaded(self, model_name):
        downloaded = self.get_downloaded_models()
        model_short = model_name.split(":")[0]
        return any(model == model_name or model.startswith(model_short) for model in downloaded)

    def check_ollama_installed(self):
        if self._ollama_installed is not None:
            return self._ollama_installed
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, timeout=5)
            self._ollama_installed = (result.returncode == 0)
        except Exception:
            self._ollama_installed = False
        return self._ollama_installed

    def check_ollama_running(self):
        try:
            resp = requests.get('http://localhost:11434/api/tags', timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def pull_model(self, model_name, progress_callback=None):
        if not self.check_ollama_installed():
            if progress_callback:
                progress_callback("❌ Ollama CLI is not installed.")
            return False
        if not self.check_ollama_running():
            if progress_callback:
                progress_callback("❌ Ollama service not running (try: `ollama serve`).")
            return False

        try:
            if progress_callback:
                progress_callback(f"Downloading {model_name}... (this may take a while)")
            proc = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line and progress_callback:
                    try:
                        msg = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        msg = line.decode('latin-1', errors='replace').strip()
                    progress_callback(msg)
            proc.wait(timeout=1200)
            success = (proc.returncode == 0)
            self._downloaded_models = None
            if not self.is_model_downloaded(model_name):
                if progress_callback:
                    progress_callback(f"Model {model_name} did not appear in the list after download.")
                return False
            if progress_callback:
                progress_callback(f"✅ {model_name} ready!")
            return success
        except subprocess.TimeoutExpired:
            if progress_callback:
                progress_callback(f"❌ Download timed out for {model_name}")
            return False
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error downloading {model_name}: {e}")
            return False

class ChloeAssistant:
    def __init__(self, profile, model=None, ollama_ok=True):
        self.model_manager = ModelManager()
        self.profile = profile
        self.current_model = model or profile.get('model')
        self.conversation_history = load_history()
        self.in_crisis_mode = False
        self.crisis_resource_shown = False
        self.ollama_ok = ollama_ok

    def bubble_sort_history_by_mood(self):
        """ Bubble Sort the history based on the mood of the messages sent by the user.

        conversation history before: [
            {'user': 'hi', 'mood': {'happiness_level': 2}},
            {'user': 'hello', 'mood': {'happiness_level': 9}},
            {'user': 'meh', 'mood': {'happiness_level': 5}}
        ]
        
        >>>bubble_sort_history_by_mood()
        >>>[{'user': 'hello', 'mood': {'happiness_level': 9}}, {'user': 'meh', 'mood': {'happiness_level': 5}}, {'user': 'hi', 'mood': {'happiness_level': 2}}]
        """
        
        if not self.conversation_history:
            return []
        history_copy = self.conversation_history.copy()
        n = len(history_copy)
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                happiness1 = history_copy[j].get('mood', {}).get('happiness_level', 5)
                happiness2 = history_copy[j + 1].get('mood', {}).get('happiness_level', 5)
                if happiness1 < happiness2:
                    history_copy[j], history_copy[j + 1] = history_copy[j + 1], history_copy[j]
                    swapped = True
            if not swapped:
                break
        return history_copy

    def linear_search_messages(self, search_term):
        """
        Linear Search for messages containing the search term in either the 'user' or 'assistant' fields.
        
        conversation history: [
        {'user': 'hi', 'assistant': 'hello', 'timestamp': '2024-06-08 10:01', 'mood': {'happiness_level': 2}},
        {'user': 'are you there?', 'assistant': 'yes, I am here!', 'timestamp': '2024-06-08 10:02', 'mood': {'happiness_level': 5}},
        {'user': 'bye', 'assistant': 'goodbye', 'timestamp': '2024-06-08 10:03', 'mood': {'happiness_level': 9}}
        ]
        
        >>> linear_search_messages('here')
        >>> [
            {'index': 1, 'type': 'user', 'content': 'are you there?', 'timestamp': '2024-06-08 10:02', 'mood': {'happiness_level': 5}},
            {'index': 1, 'type': 'assistant', 'content': 'yes, I am here!', 'timestamp': '2024-06-08 10:02', 'mood': {'happiness_level': 5}}
        ]
        """

        if not self.conversation_history or not search_term:
            return []
        search_term = search_term.lower()
        matching_messages = []
        for i, message in enumerate(self.conversation_history):
            if search_term in message.get('user', '').lower():
                matching_messages.append({
                    'index': i,
                    'type': 'user',
                    'content': message.get('user', ''),
                    'timestamp': message.get('timestamp', ''),
                    'mood': message.get('mood', {})
                })
            if search_term in message.get('assistant', '').lower():
                matching_messages.append({
                    'index': i,
                    'type': 'assistant',
                    'content': message.get('assistant', ''),
                    'timestamp': message.get('timestamp', ''),
                    'mood': message.get('mood', {})
                })
        return matching_messages

    def generate_checkin_decision(self, idle_time_sec):
        """Ask LLM if Chloe should check in, what to say, and when."""
        if not self.ollama_ok or not self.current_model:
            # Fallback: check in after 3 minutes idle, generic message
            if idle_time_sec > 180:
                return {"should_check_in": True, "wait_seconds": 0, "message": "Hey, just checking in! How are you feeling?"}
            else:
                return {"should_check_in": False}
        # Prepare context window
        recent = self.conversation_history[-6:] if self.conversation_history else []
        context = ""
        for msg in recent:
            context += f"User: {msg['user']}\nChloe: {msg['assistant']}\n"
        prompt = (
            "You are Chloe, a caring mental health companion. "
            "Given the recent conversation, the user's last detected mood, and the number of seconds since the user last sent a message, "
            "decide if you should send a check-in message, what that message should be, and after how many more seconds to wait before checking in. "
            "Reply in valid JSON only, format:\n"
            "{\"should_check_in\": true/false, \"wait_seconds\": (integer), \"message\": (if checking in, your check-in message)}\n\n"
            f"Recent conversation:\n{context}\n"
            f"Seconds since user last messaged: {int(idle_time_sec)}\n"
            "JSON:"
        )
        try:
            resp = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': self.current_model, 'prompt': prompt, 'stream': False}, timeout=20
            )
            if resp.status_code == 200:
                raw = resp.json().get('response', '')
                # Extract JSON
                start = raw.find('{')
                end = raw.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
        except Exception as e:
            print(f"Check-in LLM error: {e}")
        return {"should_check_in": False}


    def generate_age_joke(self, age):
        if self.ollama_ok and self.current_model:
            try:
                prompt = (
                    f"Come up with a light-hearted, original, funny joke or pun about someone being {age} years old. "
                    "The joke should be friendly and not about getting old or dying. Return only the joke, no intro."
                )
                resp = requests.post(
                    'http://localhost:11434/api/generate',
                    json={'model': self.current_model, 'prompt': prompt, 'stream': False}, timeout=10
                )
                if resp.status_code == 200:
                    joke = resp.json().get('response', '').strip()
                    joke = re.sub(r'^["“”\']+|["“”\']+$', '', joke)
                    return joke
            except Exception:
                pass
        fallback_jokes = [
            f"{age}? That's not an age, that's a high score!",
            f"{age} years young—nice! Still not too late to become a rockstar.",
            f"Wow, {age}! Chloe officially declares you awesome.",
            f"{age}—Prime age for world domination (or at least for pizza).",
            f"At {age}, you officially qualify as super cool!"
        ]
        return fallback_jokes[age % len(fallback_jokes)]

    def _is_valid_name(self, name):
        """Validate if a string is a reasonable first name."""
        if not name or len(name) < 2 or len(name) > 20:
            return False
        
        # Must contain only letters, hyphens, and apostrophes
        if not re.match(r"^[A-Za-z]([A-Za-z\-'])*[A-Za-z]$|^[A-Za-z]$", name):
            return False
        
        # Common words that aren't names
        common_words = {
            'the', 'and', 'but', 'yes', 'no', 'hi', 'hello', 'hey', 'what', 'how', 
            'when', 'where', 'why', 'who', 'this', 'that', 'they', 'them', 'their',
            'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can', 'may',
            'name', 'call', 'called', 'am', 'is', 'are', 'was', 'were', 'been',
            'my', 'me', 'i', 'you', 'your', 'yours', 'mine'
        }
        
        if name.lower() in common_words:
            return False
        
        return True

    def extract_name(self, text):
        # Accept any single word that only has letters, hyphens, or apostrophes (e.g. Malaz, O'Neill, Jean-Luc, Mr. Habib, etc)
        match = re.match(r"^[A-Za-z][A-Za-z\-']{0,19}$", text.strip())
        if match:
            return text.strip()
        # If LLM is available, try to extract, otherwise fallback:
        if self.ollama_ok and self.current_model:
            prompt = (
                "Extract ONLY the person's first name from this message. "
                "Return just the name (letters only), with NO extra words, punctuation, or explanation. "
                "If no valid name is found, return NONE.\n"
                f"Message: \"{text}\"\n"
                "First name only:"
            )
            try:
                resp = requests.post(
                    'http://localhost:11434/api/generate',
                    json={'model': self.current_model, 'prompt': prompt, 'stream': False}, 
                    timeout=10
                )
                if resp.status_code == 200:
                    raw = resp.json().get('response', '').strip()
                    name = re.sub(r'[^A-Za-z\-\' ]', '', raw).strip()
                    # Take first word if more than one
                    name = name.split()[0] if name else ""
                    if name.upper() in {"NONE", "NULL", "NAME", "RETURNED"}:
                        return ""
                    if re.match(r"^[A-Za-z][A-Za-z\-']{0,19}$", name):
                        return name
            except Exception:
                pass
        # Fallback: just grab first word of 2–20 letters/hyphens/apostrophes
        words = re.findall(r"\b[A-Za-z\-']{2,20}\b", text)
        return words[0] if words else ""


    def extract_age(self, text):
        """Extract age from text with improved validation."""
        if self.ollama_ok and self.current_model:
            prompt = (
                "Extract ONLY the age as a number from this message. "
                "Return just the number with no extra text or explanation. "
                "If no valid age is found, return 'NONE'.\n"
                f"Message: \"{text}\"\n"
                "Age (number only):"
            )
            try:
                resp = requests.post(
                    'http://localhost:11434/api/generate',
                    json={'model': self.current_model, 'prompt': prompt, 'stream': False}, 
                    timeout=10
                )
                if resp.status_code == 200:
                    raw = resp.json().get('response', '').strip()
                    # Extract just the number
                    numbers = re.findall(r'\d+', raw)
                    if numbers and raw.upper() != 'NONE':
                        age = int(numbers[0])
                        if 10 <= age <= 120:  # Reasonable age range
                            return age
            except Exception:
                pass
        
        # Fallback: Look for numbers in the text
        # First try to find a standalone number
        standalone_numbers = re.findall(r'\b(\d{1,3})\b', text)
        for num_str in standalone_numbers:
            try:
                age = int(num_str)
                if 10 <= age <= 120:
                    return age
            except ValueError:
                continue
        
        # If no standalone numbers, look for any digits
        all_numbers = re.findall(r'\d+', text)
        for num_str in all_numbers:
            try:
                age = int(num_str)
                if 10 <= age <= 120:
                    return age
            except ValueError:
                continue
        
        return None

    def classify_mood(self, text):
        if not self.ollama_ok or not self.current_model:
            return {'mood': 'neutral', 'happiness_level': 5, 'crisis_score': 0}
        prompt = (
            "You are an expert mental health support assistant. "
            "Analyze the following user message and output only valid JSON in this exact format: "
            '{"mood": "positive/negative/neutral/crisis", "happiness_level": 0-10, "crisis_score": 0-1}. '
            "Classify 'crisis' ONLY if the user is expressing suicidal ideation or urgent risk. "
            "Otherwise, choose positive, negative, or neutral. Here is the message: "
            f'"{text}"\nJSON:'
        )
        try:
            resp = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': self.current_model, 'prompt': prompt, 'stream': False}, timeout=10
            )
            if resp.status_code == 200:
                raw = resp.json().get('response', '')
                start = raw.find('{')
                end = raw.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
        except Exception:
            pass
        return {'mood': 'neutral', 'happiness_level': 5, 'crisis_score': 0}

    def generate_system_prompt(self, mood):

        user_age = self.profile.get('age', 25)
        base = (
            f"You are Chloe, a professional mental health associate. "
            f"Your job is to show genuine care, warmth, and empathy for the user ({self.profile.get('name','the user')}). "
            f"You never mention being an AI or assistant. "
            f"When the user is sad or negative, always reassure them that you care and will stay with them, and check in on how they're feeling. "
            f"In moments of crisis, keep your responses short and focused, never refuse to talk, never disconnect, and always try to calm and support the user with gentle, practical grounding. "
            f"If the mood is good, you can use a more relaxed, friendly style. "
            f"In all cases, your primary goal is to help the user feel heard, valued, and safe. "
            f"Always respond in 1-2 sentences only per message, unless if the user requires more detail."
        )

        # Add age-based personalization
        if user_age <= 20:
            base += (
                "\nWhen talking to someone young (like a teen), use casual tone, abbreviations, and common Gen Z slang when appropriate."
                "Keep it chill, like texting a friend. Use short, punch sentences. Example: 'lol', 'brb', 'idk', 'tbh', 'fr', 'u good?', 'nah that's crazy'. "
            )
        else:
            base += (
                "\nWhen talking to adults, be friendly, supportive, and concise. Use warm, caring language, but STRICTLY no slang. Short answers only."
            )

        if mood.get("crisis_score", 0) > 0:
            self.in_crisis_mode = True
            return base + (
                "\n\nIMPORTANT: The user may be in emotional crisis. Respond with short, calming, supportive messages. Gently ask what's going on, provide practical support (grounding techniques, breathing), and reassure them that you care. "
                "Never disconnect. If appropriate, you can provide crisis resources, but only *after* showing support and never as your only response."
            )
        elif mood.get("happiness_level", 5) <= 3 or mood.get("mood", "") == "negative":
            return base + (
                "\n\nIMPORTANT: The user is feeling down. Always reassure them that you genuinely care, offer emotional support, and check in on how they're doing."
            )
        else:
            self.in_crisis_mode = False
            return base + (
                "\nContinue the conversation naturally and warmly. Be friendly and caring."
            )

    def query_model(self, prompt, mood):
        if not self.ollama_ok or not self.current_model:
            return "LLM features are unavailable (Ollama not running)."
        try:
            system_prompt = self.generate_system_prompt(mood)
            context = ""
            if self.conversation_history:
                recent = self.conversation_history[-3:]
                context = "\n".join([f"User: {msg['user']}\nChloe: {msg['assistant']}" for msg in recent])
            full_prompt = f"System: {system_prompt}\n\nPrevious context:\n{context}\n\nUser: {prompt}\n\nChloe:"
            resp = requests.post('http://localhost:11434/api/generate', json={'model': self.current_model, 'prompt': full_prompt, 'stream': False}, timeout=1000)
            if resp.status_code == 200:
                result = resp.json().get('response', '').strip()
                if mood['crisis_score'] > 0 and self.in_crisis_mode:
                    if not self.crisis_resource_shown:
                        crisis_resources = (
                            "\n\nIf you need immediate help, here are crisis resources:"
                            "\n• US: 988 (Suicide & Crisis Lifeline)"
                            "\n• UK: 116 123 (Samaritans)"
                            "\n• Canada: 1-833-456-4566"
                            "\n• Australia: 13 11 14 (Lifeline)"
                            "\n• Emergency: 911/999/000"
                        )
                        result += crisis_resources
                        self.crisis_resource_shown = True
                elif mood['crisis_score'] == 0 and self.in_crisis_mode:
                    self.crisis_resource_shown = False
                    self.in_crisis_mode = False

                result = resp.json().get('response', '').strip()
                result = strip_action_tags(result)

                self.conversation_history.append({'user': prompt, 'assistant': result, 'mood': mood, 'timestamp': str(datetime.now())})
                save_history(self.conversation_history)
                return result
            return "I'm having trouble connecting. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"

class ModernChloeGUI:
    def __init__(self, ollama_ok=True):
        # Core initialization
        self.ollama_ok = ollama_ok
        self.profile = self.load_profile() or {}
        self.onboarding_state = None
        self.conversation_history = []
        
        
        self.chloe = ChloeAssistant(self.profile)

        # UI state
        self.current_mood = {"mood": "neutral", "happiness_level": 5, "crisis_score": 0}
        self.typing_animation_active = False
        
        # Auto-message system
        self.awaiting_user_reply = False
        self.last_user_message_time = time.time()
        self.auto_message_interval = 120  # 2 minutes
        self.proactive_enabled = True
        
        # Colors and styling
        self.colors = {
            'bg_primary': '#0f1419',
            'bg_secondary': '#1a1f2e',
            'bg_tertiary': '#252a3a',
            'accent_primary': '#64ffda',
            'accent_secondary': '#ff6b9d',
            'text_primary': '#ffffff',
            'text_secondary': '#b0bec5',
            'text_muted': '#78909c',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336',
            'user_bubble': '#2196f3',
            'chloe_bubble': '#9c27b0'
        }

        self.model_manager = ModelManager()
        self.model_categories = self.model_manager.MODELS
        self.current_model_category = None

        self.ensure_tiny_llm()

        self.setup_window()
        self.setup_styles()
        self.setup_widgets()

        
        if self.profile.get('model'):
            self.set_and_load_model(self.profile['model'])

        self.check_onboarding_status()
        self.start_background_threads()

    def ensure_tiny_llm(self):
        tiny_model = "llama3.2"
        mm = self.model_manager
        if not mm.is_model_downloaded(tiny_model):
            popup = tk.Toplevel(self.root)
            popup.title("Downloading AI Model")
            popup.geometry("400x150")
            popup.configure(bg=self.colors['bg_secondary'])
            label = tk.Label(popup, text="First time setup:\nDownloading a small AI model...", 
                            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                            font=('Segoe UI', 12))
            label.pack(expand=True, pady=30)
            self.root.update()
            def do_download():
                mm.pull_model(tiny_model, progress_callback=lambda m: None)
                # Use `after` to destroy popup in main thread
                self.root.after(0, popup.destroy)
                self.chloe.current_model = tiny_model
            threading.Thread(target=do_download, daemon=True).start()
            popup.grab_set()
            self.root.wait_window(popup)
        else:
            self.chloe.current_model = tiny_model

    def load_profile(self) -> Dict:
        """Load user profile from file"""
        try:
            with open('chloe_profile.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_profile(self):
        """Save user profile to file"""
        try:
            with open('chloe_profile.json', 'w') as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {e}")
    
    def setup_window(self):
        """Create and configure the main window"""
        self.root = tk.Tk()
        self.root.title("Chloe - Your AI Mental Health Companion")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Center window on screen
        self.center_window()
        
        # Create menu bar
        self.create_menu_bar()
        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_styles(self):
        """Configure modern styling"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure('Modern.TFrame', 
                           background=self.colors['bg_secondary'],
                           relief='flat')
        
        self.style.configure('Header.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_primary'],
                           font=('Segoe UI', 18, 'bold'))
        
        self.style.configure('Subtitle.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_secondary'],
                           font=('Segoe UI', 10))
        
        self.style.configure('Modern.TButton',
                           background=self.colors['accent_primary'],
                           foreground=self.colors['bg_primary'],
                           font=('Segoe UI', 10, 'bold'),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('Modern.TButton',
                      background=[('active', self.colors['accent_secondary']),
                                ('pressed', self.colors['accent_primary'])])
    
    def show_sorted_history_by_mood(self):
        # Use ChloeAssistant's bubble sort method
        if not hasattr(self.chloe, "bubble_sort_history_by_mood"):
            messagebox.showinfo("Not Available", "Sorting function not available.")
            return

        sorted_history = self.chloe.bubble_sort_history_by_mood()
        if not sorted_history:
            messagebox.showinfo("No Data", "No conversation history to show.")
            return

        window = tk.Toplevel(self.root)
        window.title("Conversations Sorted by Mood")
        window.geometry("800x600")
        window.configure(bg=self.colors['bg_primary'])

        text_widget = scrolledtext.ScrolledText(window, bg=self.colors['bg_secondary'], fg=self.colors['text_primary'], font=('Segoe UI', 10))
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)

        for i, msg in enumerate(sorted_history, 1):
            user = msg.get('user', '')
            assistant = msg.get('assistant', '')
            mood = msg.get('mood', {})
            ts = msg.get('timestamp', '')
            moodstr = f"Mood: {mood.get('mood', 'neutral')}, Happiness: {mood.get('happiness_level', 5)}/10"
            text_widget.insert(tk.END, f"{i}. [{ts}]\nYou: {user}\nChloe: {assistant}\n{moodstr}\n{'-'*40}\n")

        text_widget.config(state='disabled')

    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root, bg=self.colors['bg_secondary'], 
                        fg=self.colors['text_primary'], activebackground=self.colors['accent_primary'])
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'], 
                            fg=self.colors['text_primary'])
        file_menu.add_command(label="Export Chat History", command=self.export_chat)
        file_menu.add_command(label="Import Chat History", command=self.import_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Reset All Data", command=self.reset_all_data)
        menubar.add_cascade(label="File", menu=file_menu)

        # Assuming self.model_manager = self.chloe.model_manager
        model_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'], fg=self.colors['text_primary'])
        for key, group in self.model_manager.MODELS.items():
            group_name = group.get("name", key.capitalize())
            models = group.get("models", [])
            sub_menu = tk.Menu(model_menu, tearoff=0, bg=self.colors['bg_secondary'], fg=self.colors['text_primary'])
            for model in models:
                sub_menu.add_command(
                    label=model,
                    command=lambda m=model: self.set_and_load_model(m)
                )
            model_menu.add_cascade(label=group_name, menu=sub_menu)
        menubar.add_cascade(label="Model", menu=model_menu)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'], 
                                fg=self.colors['text_primary'])
        analysis_menu.add_command(label="Mood Analysis", command=self.show_mood_analysis)
        analysis_menu.add_command(label="Search Conversations (Linear Search)", command=self.search_conversations)
        analysis_menu.add_command(label="Mental Health Insights", command=self.show_insights)
        analysis_menu.add_command(label="Sort by Mood (Bubble Sort)", command=self.show_sorted_history_by_mood)

        menubar.add_cascade(label="Analysis", menu=analysis_menu)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'], 
                                fg=self.colors['text_primary'])
        settings_menu.add_command(label="Preferences", command=self.show_preferences)
        settings_menu.add_command(label="Model Settings", command=self.show_model_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)

    def set_and_load_model(self, model_name):
        self.status_var.set(f"Preparing to load model: {model_name}")
        self.root.update_idletasks()

        if not self.ollama_ok or not hasattr(self.chloe, 'model_manager'):
            self.status_var.set("Ollama is not available or not running.")
            return

        def after_load(success):
            if success:
                self.chloe.current_model = model_name
                self.profile['model'] = model_name
                self.save_profile()
                self.status_var.set(f"Model loaded: {model_name}")
                self.model_status_label.config(text=f"{model_name} ready", foreground=self.colors['success'])
            else:
                self.status_var.set(f"Failed to load model: {model_name}")

        def do_load():
            mm = self.chloe.model_manager
            # Check if model is downloaded
            if not mm.is_model_downloaded(model_name):
                self.status_var.set(f"Downloading {model_name} via Ollama...")
                self.root.update_idletasks()
                success = mm.pull_model(model_name, progress_callback=lambda msg: self.status_var.set(msg))
                self.root.after(0, lambda: after_load(success))
            else:
                self.root.after(0, lambda: after_load(True))

        threading.Thread(target=do_load, daemon=True).start()

    
    def setup_widgets(self):
        """Create and arrange all UI widgets"""
        # Main container
        main_container = ttk.Frame(self.root, style='Modern.TFrame')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header section
        self.create_header(main_container)
        
        # Status and model section
        self.create_status_section(main_container)
        
        # Chat section
        self.create_chat_section(main_container)
        
        # Input section
        self.create_input_section(main_container)
        
        # Footer
        self.create_footer(main_container)
    
    def create_header(self, parent):
        """Create the header section"""
        header_frame = ttk.Frame(parent, style='Modern.TFrame')
        header_frame.pack(fill='x', pady=(0, 15))
        
        # Left side - title and greeting
        left_frame = ttk.Frame(header_frame, style='Modern.TFrame')
        left_frame.pack(side='left', fill='x', expand=True)
        
        self.title_label = ttk.Label(left_frame, text="🤖 Chloe AI", style='Header.TLabel')
        self.title_label.pack(anchor='w')
        
        greeting_text = self.get_greeting_text()
        self.greeting_label = ttk.Label(left_frame, text=greeting_text, style='Subtitle.TLabel')
        self.greeting_label.pack(anchor='w')
        
        # Right side - mood and status
        right_frame = ttk.Frame(header_frame, style='Modern.TFrame')
        right_frame.pack(side='right')
        
        self.mood_display = self.create_mood_widget(right_frame)
        self.mood_display.pack(anchor='e')
    
    def get_greeting_text(self) -> str:
        """Generate personalized greeting"""
        name = self.profile.get('name', '')
        hour = datetime.now().hour
        
        if hour < 12:
            time_greeting = "Good morning"
        elif hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        if name:
            return f"{time_greeting}, {name}! Ready to chat?"
        else:
            return f"{time_greeting}! Let's get started."
    
    def create_mood_widget(self, parent):
        """Create the mood display widget"""
        mood_frame = ttk.Frame(parent, style='Modern.TFrame')
        
        # Mood emoji and text
        self.mood_emoji = ttk.Label(mood_frame, text="😐", 
                                   background=self.colors['bg_primary'],
                                   font=('Segoe UI', 16))
        self.mood_emoji.pack()
        
        self.mood_text = ttk.Label(mood_frame, text="Neutral", 
                                  background=self.colors['bg_primary'],
                                  foreground=self.colors['text_secondary'],
                                  font=('Segoe UI', 9))
        self.mood_text.pack()
        
        return mood_frame
    
    def create_status_section(self, parent):
        """Create status and model selection section"""
        status_frame = ttk.Frame(parent, style='Modern.TFrame')
        status_frame.pack(fill='x', pady=(0, 10))
        
        # Model status
        model_frame = ttk.LabelFrame(status_frame, text=" AI Model Status ", 
                                    style='Modern.TFrame')
        model_frame.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        self.model_status_label = ttk.Label(model_frame, 
                                           text="Checking model status...",
                                           background=self.colors['bg_secondary'],
                                           foreground=self.colors['text_secondary'])
        self.model_status_label.pack(padx=10, pady=5)
        
        # Connection status
        connection_frame = ttk.LabelFrame(status_frame, text=" Connection ", 
                                         style='Modern.TFrame')
        connection_frame.pack(side='right')
        
        self.connection_status = ttk.Label(connection_frame,
                                          text="Connecting...",
                                          background=self.colors['bg_secondary'],
                                          foreground=self.colors['warning'])
        self.connection_status.pack(padx=10, pady=5)
    
    def create_chat_section(self, parent):
        """Create the main chat area"""
        chat_frame = ttk.Frame(parent, style='Modern.TFrame')
        chat_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Create custom chat display
        self.chat_canvas = tk.Canvas(chat_frame, 
                                    bg=self.colors['bg_primary'],
                                    highlightthickness=0)
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, orient='vertical', 
                                           command=self.chat_canvas.yview)
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        
        self.chat_container = ttk.Frame(self.chat_canvas, style='Modern.TFrame')
        self.chat_canvas.create_window((0, 0), window=self.chat_container, anchor='nw')
        
        self.chat_canvas.pack(side='left', fill='both', expand=True)
        self.chat_scrollbar.pack(side='right', fill='y')
        
        # Bind canvas resize
        self.chat_container.bind('<Configure>', self.on_chat_configure)
        self.chat_canvas.bind('<Configure>', self.on_canvas_configure)
    
        self.chat_container.bind('<Enter>', lambda e: self._bind_mousewheel(self.chat_container))
        self.chat_container.bind('<Leave>', lambda e: self._unbind_mousewheel(self.chat_container))

        
        # Mouse wheel scrolling
        self.chat_canvas.bind("<MouseWheel>", self.on_mousewheel)
    
    def create_input_section(self, parent):
        """Create the message input area"""
        input_frame = ttk.Frame(parent, style='Modern.TFrame')
        input_frame.pack(fill='x', pady=(0, 10))
        
        # Text input with modern styling
        self.input_text = tk.Text(input_frame,
                                 height=3,
                                 font=('Segoe UI', 11),
                                 bg=self.colors['bg_tertiary'],
                                 fg=self.colors['text_primary'],
                                 insertbackground=self.colors['accent_primary'],
                                 selectbackground=self.colors['accent_primary'],
                                 relief='flat',
                                 padx=15,
                                 pady=10,
                                 wrap='word')
        self.input_text.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Send button
        button_frame = ttk.Frame(input_frame, style='Modern.TFrame')
        button_frame.pack(side='right', fill='y')
        
        self.send_button = ttk.Button(button_frame, text="Send ➤", 
                                     style='Modern.TButton',
                                     command=self.send_message)
        self.send_button.pack(fill='both', expand=True)
        
        # Bind enter key
        self.input_text.bind('<Return>', self.on_enter_key)
        self.input_text.bind('<Shift-Return>', self.on_shift_enter)
        
        # Placeholder text
        self.add_placeholder()
    
    def add_placeholder(self):
        """Add placeholder text to input field"""
        placeholder = "Type your message here... (Press Enter to send, Shift+Enter for new line)"
        self.input_text.insert('1.0', placeholder)
        self.input_text.config(fg=self.colors['text_muted'])
        
        def on_focus_in(event):
            if self.input_text.get('1.0', 'end-1c') == placeholder:
                self.input_text.delete('1.0', tk.END)
                self.input_text.config(fg=self.colors['text_primary'])
        
        def on_focus_out(event):
            if not self.input_text.get('1.0', 'end-1c').strip():
                self.input_text.insert('1.0', placeholder)
                self.input_text.config(fg=self.colors['text_muted'])
        
        self.input_text.bind('<FocusIn>', on_focus_in)
        self.input_text.bind('<FocusOut>', on_focus_out)
    
    def create_footer(self, parent):
        """Create the footer status bar"""
        footer_frame = ttk.Frame(parent, style='Modern.TFrame')
        footer_frame.pack(fill='x')
        
        # Status message
        self.status_var = tk.StringVar(value="Welcome to Chloe AI")
        self.status_label = ttk.Label(footer_frame, textvariable=self.status_var,
                                     background=self.colors['bg_secondary'],
                                     foreground=self.colors['text_secondary'],
                                     font=('Segoe UI', 9))
        self.status_label.pack(side='left', padx=5)
        
        # Typing indicator
        self.typing_indicator = ttk.Label(footer_frame, text="",
                                         background=self.colors['bg_secondary'],
                                         foreground=self.colors['accent_primary'],
                                         font=('Segoe UI', 9, 'italic'))
        self.typing_indicator.pack(side='right', padx=5)
    
    def check_onboarding_status(self):
        """Check if user needs onboarding"""
        if not self.profile.get('name'):
            self.onboarding_state = 'name'
        elif not self.profile.get('age'):
            self.onboarding_state = 'age'
        elif not self.profile.get('preferences_set'):
            self.onboarding_state = 'preferences'
        
        if self.onboarding_state:
            self.start_onboarding()
        else:
            self.load_chat_history()
            self.show_welcome_message()
    
    def start_onboarding(self):
        """Begin the onboarding process"""
        if self.onboarding_state == 'name':
            self.add_message("Chloe", 
                           "Hi there! I'm Chloe, your AI mental health companion. "
                           "I'm here to listen, support, and help you navigate life's challenges.\n\n"
                           "Let's start by getting to know each other. What should I call you? "
                           "(Just your first name is perfect!)")
        elif self.onboarding_state == 'age':
            name = self.profile.get('name', '')
            self.add_message("Chloe",
                           f"Nice to meet you, {name}!\n\n"
                           f"To provide you with the most appropriate support, "
                           f"could you tell me your age? This helps me tailor our conversations better.")
        elif self.onboarding_state == 'preferences':
            self.show_preferences_setup()
    
    def show_welcome_message(self):
        """Show welcome message for returning users"""
        name = self.profile.get('name', 'there')
        messages = [
            f"Welcome back, {name}! 🌟",
            "I'm here and ready to chat whenever you need support.",
            "How are you feeling today?"
        ]
        
        for i, msg in enumerate(messages):
            self.root.after(i * 1500, lambda m=msg: self.add_message("Chloe", m))
    
    def start_background_threads(self):
        """Start background processes"""
        # Proactive messaging thread
        self.proactive_thread = threading.Thread(target=self.proactive_message_loop, daemon=True)
        self.proactive_thread.start()
        
        # Status checking thread
        self.status_thread = threading.Thread(target=self.status_check_loop, daemon=True)
        self.status_thread.start()
    
    def proactive_message_loop(self):
        """Background loop for proactive messaging"""
        while True:
            try:
                if (self.proactive_enabled and 
                    not self.onboarding_state and 
                    not self.awaiting_user_reply):
                    
                    idle_time = time.time() - self.last_user_message_time
                    
                    if idle_time > self.auto_message_interval:
                        message = self.generate_proactive_message(idle_time)
                        if message:
                            self.root.after(0, lambda: self.add_message("Chloe", message))
                            self.awaiting_user_reply = True
                            self.last_user_message_time = time.time()
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Proactive message error: {e}")
                time.sleep(60)
    
    def status_check_loop(self):
        """Background loop for status checking"""
        while True:
            try:
                # Check model status
                if hasattr(self, 'chloe') and hasattr(self.chloe, 'current_model') and self.chloe.current_model:
                    model_text = f"✅ {self.chloe.current_model} ready"
                    self.root.after(0, lambda: self.model_status_label.config(text=model_text,
                                                                             foreground=self.colors['success']))
                else:
                    model_text = "❌ No model loaded"
                    self.root.after(0, lambda: self.model_status_label.config(text=model_text,
                                                                             foreground=self.colors['error']))
                
                # Check connection
                if self.ollama_ok:
                    conn_text = "🟢 Connected"
                    color = self.colors['success']
                else:
                    conn_text = "🔴 Offline Mode"
                    color = self.colors['error']
                
                self.root.after(0, lambda: self.connection_status.config(text=conn_text, foreground=color))
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"Status check error: {e}")
                time.sleep(30)
    
    def generate_proactive_message(self, idle_time: float) -> Optional[str]:
        """Generate a proactive check-in message"""
        if idle_time < 300:  # 5 minutes
            return None
        
        messages = [
            "How are you doing? I'm here if you need to talk about anything.",
            "Just checking in! Sometimes it helps to share what's on your mind.",
            "I noticed it's been quiet for a while. Want to chat about how your day went?",
            "Thinking of you! How are you feeling right now?",
            "Hi there! I'm here if you'd like to talk or just need someone to listen.",
        ]
        
        # Select based on current mood and time of day
        return random.choice(messages)
    
    def on_enter_key(self, event):
        """Handle Enter key press"""
        self.send_message()
        return 'break'
    
    def on_shift_enter(self, event):
        """Handle Shift+Enter (new line)"""
        return None
    
    def send_message(self):
        """Send user message and get AI response"""
        message = self.input_text.get('1.0', 'end-1c').strip()
        placeholder = "Type your message here... (Press Enter to send, Shift+Enter for new line)"
        
        if not message or message == placeholder:
            return
        
        # Clear input
        self.input_text.delete('1.0', tk.END)
        
        # Add user message
        self.add_message("You", message)
        
        # Update state
        self.last_user_message_time = time.time()
        self.awaiting_user_reply = False
        
        # Handle onboarding
        if self.onboarding_state:
            self.handle_onboarding_response(message)
            return
        
        # Generate Chloe's reply IN A NEW THREAD, so you can send another message instantly
        threading.Thread(target=lambda: self.process_user_message(message), daemon=True).start()
    
    def handle_onboarding_response(self, message: str):
        if self.onboarding_state == 'name':
            name = self.extract_name(message)
            if name:
                self.profile['name'] = name
                self.save_profile()
                self.onboarding_state = 'age'
                self.greeting_label.config(text=self.get_greeting_text())
                self.add_message("Chloe", f"Great to meet you, {name}!\n\n"
                            f"Now, could you tell me your age? This helps me provide "
                            f"more appropriate support and conversation.")
            else:
                self.add_message(
                    "Chloe",
                    "Sorry, I couldn't detect a real first name. Please type ONLY your first name (letters only, e.g., 'Alex'). No numbers or special characters."
                )
                # Stay in 'name' state, user will be prompted again.

        elif self.onboarding_state == 'age':
            age = self.extract_age(message)
            if age and 5 <= age <= 120:
                self.profile['age'] = age
                self.save_profile()
                self.onboarding_state = None

                response = self.chloe.generate_age_joke(age)
                self.add_message("Chloe", f"Thanks! {response}\n\nNow, to access smarter or faster versions of Chloe, please go to the Model tab at the top, choose your preferred AI model, and click to load it! Otherwise, you can just continue chatting as is.")
                self.profile['preferences_set'] = True
                self.save_profile()
            else:
                self.add_message(
                    "Chloe",
                    "Sorry, I couldn't detect a valid age. Please type just your age as a number between 5 and 120 (e.g., '25')."
                )
                # Stay in 'age' state, user will be prompted again.
    
    def extract_name(self, text: str) -> Optional[str]:
        # Use LLM extraction if available
        name = None
        if hasattr(self, 'chloe') and hasattr(self.chloe, 'extract_name'):
            name = self.chloe.extract_name(text)
        return name
    
    def extract_age(self, text: str) -> Optional[int]:
        age = None
        if hasattr(self, 'chloe') and hasattr(self.chloe, 'extract_age'):
            age = self.chloe.extract_age(text)
        return age
    
    def process_user_message(self, message: str):
        """Process regular user message"""
        # Show typing indicator
        self.show_typing_indicator(True)
        
        # Analyze mood
        mood = self.analyze_mood(message)
        self.update_mood_display(mood)
        
        # Generate response in background
        def generate_response():
            try:
                response = self.get_ai_response(message, mood)
                self.root.after(0, lambda: self.handle_ai_response(response))
            except Exception as e:
                error_msg = "I'm having trouble processing that right now. Could you try again?"
                self.root.after(0, lambda: self.handle_ai_response(error_msg))
        
        threading.Thread(target=generate_response, daemon=True).start()
    
    def get_ai_response(self, message: str, mood: Dict) -> str:
        """Get AI response to user message"""
        try:
            if hasattr(self.chloe, 'query_model'):
                return self.chloe.query_model(message, mood)
            else:
                # Fallback response system
                return self.generate_fallback_response(message, mood)
        except Exception:
            return self.generate_fallback_response(message, mood)
    
    def generate_fallback_response(self, message: str, mood: Dict) -> str:
        """Generate fallback response when AI is unavailable"""
        mood_level = mood.get('mood', 'neutral')
        
        if mood_level == 'negative' or mood.get('crisis_score', 0) > 3:
            return ("I can hear that you're going through a difficult time. "
                   "While I'm having technical difficulties right now, please know that "
                   "your feelings are valid and important. If you're in crisis, please "
                   "consider reaching out to a mental health professional or crisis hotline.")
        
        responses = [
            "I appreciate you sharing that with me. How does that make you feel?",
            "That sounds important to you. Can you tell me more about it?",
            "I'm listening. What would be most helpful for you right now?",
            "Thank you for opening up. What thoughts are going through your mind about this?",
        ]
        
        return random.choice(responses)
    
    def analyze_mood(self, message: str) -> Dict:
        """Analyze mood from message"""
        # Basic mood analysis
        negative_words = ['sad', 'angry', 'depressed', 'anxious', 'worried', 'scared', 'hurt', 'pain']
        positive_words = ['happy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'joy', 'good']
        
        message_lower = message.lower()
        
        neg_count = sum(1 for word in negative_words if word in message_lower)
        pos_count = sum(1 for word in positive_words if word in message_lower)
        
        if neg_count > pos_count:
            mood = 'negative'
            happiness = max(1, 5 - neg_count)
        elif pos_count > neg_count:
            mood = 'positive'
            happiness = min(10, 5 + pos_count)
        else:
            mood = 'neutral'
            happiness = 5
        
        crisis_indicators = ['kill', 'die', 'suicide', 'hurt myself', 'end it all']
        crisis_score = sum(1 for indicator in crisis_indicators if indicator in message_lower)
        
        return {
            'mood': mood,
            'happiness_level': happiness,
            'crisis_score': crisis_score
        }
    
    def handle_ai_response(self, response: str):
        """Handle AI response"""
        self.show_typing_indicator(False)
        self.add_message("Chloe", response)
        
        # Save conversation
        self.save_conversation_turn(self.get_last_user_message(), response)
    
    def get_last_user_message(self) -> str:
        """Get the last user message"""
        # This would normally get from conversation history
        return "User message"  # Simplified for demo
    
    def save_conversation_turn(self, user_msg: str, ai_msg: str):
        """Save conversation turn to history"""
        turn = {
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'assistant': ai_msg,
            'mood': self.current_mood
        }
        self.conversation_history.append(turn)
        
        # Save to file
        try:
            with open('chloe_history.json', 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def show_typing_indicator(self, show: bool):
        """Show or hide typing indicator"""
        if show:
            self.typing_indicator.config(text="Chloe is typing...")
            self.typing_animation_active = True
            self.animate_typing_dots()
        else:
            self.typing_indicator.config(text="")
            self.typing_animation_active = False
    
    def animate_typing_dots(self):
        """Animate typing indicator dots"""
        if not self.typing_animation_active:
            return
        
        current_text = self.typing_indicator.cget('text')
        if current_text.endswith('...'):
            new_text = "Chloe is typing"
        elif current_text.endswith('..'):
            new_text = "Chloe is typing..."
        elif current_text.endswith('.'):
            new_text = "Chloe is typing.."
        else:
            new_text = "Chloe is typing."
        
        self.typing_indicator.config(text=new_text)
        self.root.after(500, self.animate_typing_dots)
    
    def update_mood_display(self, mood: Dict):
        """Update mood display widget"""
        self.current_mood = mood
        mood_type = mood.get('mood', 'neutral')
        happiness = mood.get('happiness_level', 5)
        crisis_score = mood.get('crisis_score', 0)
        
        # Update emoji
        if crisis_score > 0:
            emoji = "😰"
            mood_text = "Crisis"
            color = self.colors['error']
        elif mood_type == 'positive':
            emoji = "😊" if happiness > 7 else "🙂"
            mood_text = "Positive"
            color = self.colors['success']
        elif mood_type == 'negative':
            emoji = "😔" if happiness < 3 else "😐"
            mood_text = "Negative" 
            color = self.colors['error']
        else:
            emoji = "😐"
            mood_text = "Neutral"
            color = self.colors['text_secondary']
        
        self.mood_emoji.config(text=emoji)
        self.mood_text.config(text=f"{mood_text} ({happiness}/10)", foreground=color)
    
    def add_message(self, sender: str, message: str, animate: bool = True):
        """Add a message to the chat display"""
        # Create message bubble
        message_frame = ttk.Frame(self.chat_container, style='Modern.TFrame')
        message_frame.pack(fill='x', padx=10, pady=5)
        
        # Make every new message frame listen to scroll
        message_frame.bind('<Enter>', lambda e: self._bind_mousewheel(message_frame))
        message_frame.bind('<Leave>', lambda e: self._unbind_mousewheel(message_frame))

        # Determine alignment and colors
        if sender == "You":
            anchor = 'e'
            bg_color = self.colors['user_bubble']
            text_color = 'white'
            sender_label = "You"
        else:
            anchor = 'w' 
            bg_color = self.colors['chloe_bubble']
            text_color = 'white'
            sender_label = "🤖 Chloe"
        
        # Create bubble container
        bubble_container = tk.Frame(message_frame, bg=self.colors['bg_primary'])
        bubble_container.pack(anchor=anchor, padx=(0 if sender == "You" else 50, 50 if sender == "You" else 0))
        
        # Sender label
        sender_frame = tk.Frame(bubble_container, bg=self.colors['bg_primary'])
        sender_frame.pack(fill='x', pady=(0, 2))
        
        sender_label_widget = tk.Label(sender_frame, text=sender_label,
                                      bg=self.colors['bg_primary'],
                                      fg=self.colors['text_muted'],
                                      font=('Segoe UI', 9, 'bold'))
        sender_label_widget.pack(anchor=anchor)
        
        # Message bubble
        bubble = tk.Frame(bubble_container, bg=bg_color, relief='flat')
        bubble.pack(anchor=anchor)
        
        # Message text
        message_label = tk.Label(bubble, text=message,
                                bg=bg_color,
                                fg=text_color,
                                font=('Segoe UI', 11),
                                wraplength=400,
                                justify='left',
                                padx=15,
                                pady=10)
        message_label.pack()
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        time_label = tk.Label(bubble_container, text=timestamp,
                             bg=self.colors['bg_primary'],
                             fg=self.colors['text_muted'],
                             font=('Segoe UI', 8))
        time_label.pack(anchor=anchor, pady=(2, 0))
        
        # Update scroll region and scroll to bottom
        self.root.after(100, self.update_chat_scroll)
        
        # Animate message appearance
        if animate:
            self.animate_message_appearance(bubble)
    
    def _bind_mousewheel(self, widget):
        widget.bind_all('<MouseWheel>', self.on_mousewheel)

    def _unbind_mousewheel(self, widget):
        widget.unbind_all('<MouseWheel>')

    def animate_message_appearance(self, widget):
        """Animate message bubble appearance"""
        # Simple fade-in effect by adjusting the widget's appearance
        widget.configure(relief='raised', bd=1)
        self.root.after(200, lambda: widget.configure(relief='flat', bd=0))
    
    def update_chat_scroll(self):
        """Update chat scroll region and scroll to bottom"""
        self.chat_container.update_idletasks()
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        self.chat_canvas.yview_moveto(1.0)
    
    def on_chat_configure(self, event):
        """Handle chat container resize"""
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        canvas_width = event.width
        self.chat_canvas.itemconfig(self.chat_canvas.find_all()[0], width=canvas_width)
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling in chat"""
        self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def load_chat_history(self):
        """Load previous chat history"""
        try:
            with open('chloe_history.json', 'r') as f:
                self.conversation_history = json.load(f)
            
            if self.conversation_history:
                self.add_message("System", f"💾 Restored {len(self.conversation_history)} previous messages", animate=False)
                
                # Show last few messages
                for msg in self.conversation_history[-3:]:
                    self.add_message("You", msg.get('user', ''), animate=False)
                    self.add_message("Chloe", msg.get('assistant', ''), animate=False)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    # Menu action methods
    def export_chat(self):
        """Export chat history"""
        if not self.conversation_history:
            messagebox.showinfo("No Data", "No conversation history to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")],
            title="Export Chat History"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(self.conversation_history, f, indent=2)
                else:
                    with open(filename, 'w', encoding='utf-8') as f:
                        for msg in self.conversation_history:
                            f.write(f"[{msg.get('timestamp', '')}]\n")
                            f.write(f"You: {msg.get('user', '')}\n")
                            f.write(f"Chloe: {msg.get('assistant', '')}\n")
                            f.write("-" * 50 + "\n\n")
                
                messagebox.showinfo("Success", f"Chat history exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def import_chat(self):
        """Import chat history"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Import Chat History"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    imported_history = json.load(f)
                
                if isinstance(imported_history, list):
                    self.conversation_history = imported_history
                    
                    # Clear current chat display
                    for widget in self.chat_container.winfo_children():
                        widget.destroy()
                    
                    # Reload chat
                    self.load_chat_history()
                    messagebox.showinfo("Success", f"Imported {len(imported_history)} messages")
                else:
                    raise ValueError("Invalid file format")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import: {e}")

    def reset_all_data(self):
        """
        Force-wipe Chloe’s local state.

        1. Gracefully closes every Chroma handle we know about.
        2. Spawns a detached “janitor” which keeps retrying the delete
        (with exponential back–off) until the folder is *really* gone
        or we hit 30 s.
        3. Works on Windows / macOS / Linux – no external deps.
        """

        if not messagebox.askyesno(
            "Confirm Reset",
            "Chloe will close and ALL your local data will be PERMANENTLY deleted.\n\n"
            "Proceed?"):
            return

        # ------------------------------------------------------------------ #
        # Step 1 –  try to release file handles held by this process
        # ------------------------------------------------------------------ #
        try:
            # Chroma keeps a global singleton client; destroy it if possible
            if hasattr(chromadb, "PersistentClient"):
                try:
                    _tmp_client = chromadb.PersistentClient(path=CHROMADB_PATH)
                    _tmp_client.reset()      # closes DuckDB + ann-index
                except Exception:
                    pass
        except Exception:
            pass
        gc.collect()

        chroma_dir = os.path.abspath(CHROMADB_PATH)
        trash_tag  = f".trash_{int(time.time())}"
        json_files = [
            os.path.abspath("chloe_profile.json"),
            os.path.abspath("chloe_history.json"),
            os.path.abspath("chloe_convo.json"),
        ]
        parent_pid = os.getpid()

        # ------------------------------------------------------------------ #
        # Step 2 –  write the janitor script to a temp file
        # ------------------------------------------------------------------ #
        janitor_py = textwrap.dedent(f"""
            import os, shutil, stat, time, sys, random

            TARGET      = r\"\"\"{chroma_dir}\"\"\"
            TRASH_ALIAS = TARGET + r\"{trash_tag}\"
            JSONS       = {json_files!r}
            PARENT_PID  = {parent_pid}

            def _chmod_and_retry(func, path, _):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass

            # Wait for parent to exit (max 10 s)
            for _ in range(100):
                try:
                    os.kill(PARENT_PID, 0)
                    time.sleep(0.1)
                except OSError:
                    break  # parent is dead

            # 1) Try quick rename (works even if a file is still open)
            if os.path.exists(TARGET):
                try:
                    os.rename(TARGET, TRASH_ALIAS)
                    TARGET = TRASH_ALIAS
                except Exception:
                    pass  # rename failed; we'll just delete in place

            # 2) Robust rm - keep trying up to 30 s
            deadline = time.time() + 30
            backoff  = 0.05
            while os.path.exists(TARGET) and time.time() < deadline:
                try:
                    shutil.rmtree(TARGET, onerror=_chmod_and_retry)
                except Exception:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 2.0)  # expo back-off

            # 3) Wipe JSONs
            for f in JSONS:
                try:
                    os.remove(f)
                except PermissionError:
                    try:
                        os.chmod(f, stat.S_IWRITE)
                        os.remove(f)
                    except Exception:
                        pass
                except FileNotFoundError:
                    pass
        """)

        tmp = tempfile.NamedTemporaryFile(delete=False,
                                        suffix=".py",
                                        mode="w",
                                        encoding="utf-8")
        tmp.write(janitor_py)
        tmp.close()

        # ------------------------------------------------------------------ #
        # Step 3 – spawn janitor *detached*
        # ------------------------------------------------------------------ #
        popen_kw = dict(stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        if os.name == "nt":
            DETACHED_PROCESS      = 0x00000008
            CREATE_NEW_PROCESS_GP = 0x00000200
            popen_kw["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GP
        else:
            popen_kw["start_new_session"] = True

        subprocess.Popen([sys.executable, tmp.name], **popen_kw)

        # ------------------------------------------------------------------ #
        # Step 4 – close the GUI and bail out
        # ------------------------------------------------------------------ #
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)


    def show_mood_analysis(self):
        """Show detailed mood analysis"""
        if not self.conversation_history:
            messagebox.showinfo("No Data", "No conversation history to analyze.")
            return
        
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Mood Analysis")
        analysis_window.geometry("900x700")
        analysis_window.configure(bg=self.colors['bg_primary'])
        
        # Create notebook for different analysis views
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Overview tab
        overview_frame = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(overview_frame, text="Overview")
        
        self.create_mood_overview(overview_frame)
        
        # Timeline tab
        timeline_frame = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(timeline_frame, text="Timeline")
        
        self.create_mood_timeline(timeline_frame)
        
        # Insights tab
        insights_frame = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(insights_frame, text="Insights")
        
        self.create_mood_insights(insights_frame)
    
    def create_mood_overview(self, parent):
        """Create mood overview display"""
        # Calculate mood statistics
        moods = [msg.get('mood', {}) for msg in self.conversation_history]
        mood_types = [m.get('mood', 'neutral') for m in moods if m]
        happiness_levels = [m.get('happiness_level', 5) for m in moods if m]
        
        if not mood_types:
            ttk.Label(parent, text="No mood data available", style='Subtitle.TLabel').pack(pady=50)
            return
        
        # Statistics
        stats_frame = ttk.LabelFrame(parent, text=" Mood Statistics ")
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        avg_happiness = sum(happiness_levels) / len(happiness_levels) if happiness_levels else 5
        positive_count = mood_types.count('positive')
        negative_count = mood_types.count('negative')
        neutral_count = mood_types.count('neutral')
        
        stats_text = f"""
        Average Happiness Level: {avg_happiness:.1f}/10
        
        Mood Distribution:
        • Positive: {positive_count} ({positive_count/len(mood_types)*100:.1f}%)
        • Neutral: {neutral_count} ({neutral_count/len(mood_types)*100:.1f}%)
        • Negative: {negative_count} ({negative_count/len(mood_types)*100:.1f}%)
        
        Total Conversations: {len(self.conversation_history)}
        """
        
        stats_label = tk.Label(stats_frame, text=stats_text,
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'],
                              font=('Segoe UI', 11),
                              justify='left')
        stats_label.pack(padx=10, pady=10)
    
    def create_mood_timeline(self, parent):
        """Create mood timeline display"""
        # Simplified timeline - would normally use a plotting library
        timeline_text = scrolledtext.ScrolledText(parent,
                                                 bg=self.colors['bg_secondary'],
                                                 fg=self.colors['text_primary'],
                                                 font=('Consolas', 10))
        timeline_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        timeline_text.insert(tk.END, "📊 MOOD TIMELINE\n")
        timeline_text.insert(tk.END, "=" * 50 + "\n\n")
        
        for i, msg in enumerate(self.conversation_history[-20:], 1):  # Last 20 messages
            mood = msg.get('mood', {})
            mood_type = mood.get('mood', 'neutral')
            happiness = mood.get('happiness_level', 5)
            timestamp = msg.get('timestamp', '')
            
            mood_emoji = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(mood_type, "😐")
            
            timeline_text.insert(tk.END, f"{i:2d}. {mood_emoji} {mood_type.title()} ({happiness}/10)\n")
            timeline_text.insert(tk.END, f"    {timestamp}\n")
            timeline_text.insert(tk.END, f"    \"{msg.get('user', '')[:60]}...\"\n\n")
        
        timeline_text.config(state='disabled')
    
    def create_mood_insights(self, parent):
        """Create mood insights display"""
        insights_text = scrolledtext.ScrolledText(parent,
                                                 bg=self.colors['bg_secondary'],
                                                 fg=self.colors['text_primary'],
                                                 font=('Segoe UI', 11))
        insights_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Generate insights based on conversation history
        insights = self.generate_mood_insights()
        
        insights_text.insert(tk.END, "🧠 MENTAL HEALTH INSIGHTS\n")
        insights_text.insert(tk.END, "=" * 50 + "\n\n")
        
        for insight in insights:
            insights_text.insert(tk.END, f"• {insight}\n\n")
        
        insights_text.config(state='disabled')
    
    def generate_mood_insights(self) -> List[str]:
        """Generate insights from conversation history"""
        if not self.conversation_history:
            return ["No conversation data available for analysis."]
        
        insights = []
        
        # Analyze patterns
        moods = [msg.get('mood', {}) for msg in self.conversation_history]
        happiness_levels = [m.get('happiness_level', 5) for m in moods if m]
        
        if happiness_levels:
            avg_happiness = sum(happiness_levels) / len(happiness_levels)
            
            if avg_happiness > 7:
                insights.append("You've been maintaining a generally positive mood! Keep up the great work.")
            elif avg_happiness < 4:
                insights.append("I notice you've been experiencing some difficult emotions lately. Remember that it's okay to have tough days.")
            else:
                insights.append("Your mood has been relatively balanced, which shows good emotional stability.")
        
        # Recent trend analysis
        if len(happiness_levels) >= 5:
            recent_avg = sum(happiness_levels[-5:]) / 5
            earlier_avg = sum(happiness_levels[:-5]) / len(happiness_levels[:-5])
            
            if recent_avg > earlier_avg + 1:
                insights.append("Your mood has been improving recently - that's wonderful to see!")
            elif recent_avg < earlier_avg - 1:
                insights.append("I've noticed your mood has been lower recently. Consider reaching out for additional support if needed.")
        
        insights.append("Remember: talking about your feelings is a sign of strength, not weakness.")
        insights.append("Your mental health journey is unique to you, and every step forward matters.")
        
        return insights
    
    def search_conversations(self):
        """Search through conversations"""
        search_term = simpledialog.askstring("Search Conversations", 
                                            "Enter search term:")
        if not search_term:
            return
        
        # Search through conversation history
        results = self.chloe.linear_search_messages(search_term)

        if not results:
            messagebox.showinfo("Search Results", f"No conversations found containing '{search_term}'")
            return

        # Display results
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Search Results: '{search_term}'")
        results_window.geometry("800x600")
        results_window.configure(bg=self.colors['bg_primary'])

        results_text = scrolledtext.ScrolledText(results_window,
                                                bg=self.colors['bg_secondary'],
                                                fg=self.colors['text_primary'],
                                                font=('Segoe UI', 10))
        results_text.pack(fill='both', expand=True, padx=10, pady=10)

        results_text.insert(tk.END, f"🔍 SEARCH RESULTS FOR '{search_term}'\n")
        results_text.insert(tk.END, f"Found {len(results)} matching messages\n")
        results_text.insert(tk.END, "=" * 60 + "\n\n")

        for i, result in enumerate(results, 1):
            content = result['content']
            type_ = result['type'].capitalize()
            ts = result.get('timestamp', '')
            mood = result.get('mood', {})
            moodstr = f"Mood: {mood.get('mood', 'neutral')}, Happiness: {mood.get('happiness_level', 5)}/10"
            results_text.insert(tk.END, f"{i}. [{ts}] {type_}:\n{content}\n{moodstr}\n{'-'*40}\n\n")

        results_text.config(state='disabled')

    
    def show_insights(self):
        """Show mental health insights"""
        self.show_mood_analysis()  # For now, use the same window
    
    def show_preferences(self):
        """Show preferences dialog"""
        prefs_window = tk.Toplevel(self.root)
        prefs_window.title("Preferences")
        prefs_window.geometry("500x400")
        prefs_window.configure(bg=self.colors['bg_primary'])
        
        # Proactive messaging setting
        proactive_frame = ttk.LabelFrame(prefs_window, text=" Proactive Messaging ")
        proactive_frame.pack(fill='x', padx=10, pady=10)
        
        self.proactive_var = tk.BooleanVar(value=self.proactive_enabled)
        proactive_check = ttk.Checkbutton(proactive_frame, 
                                        text="Enable proactive check-ins",
                                        variable=self.proactive_var,
                                        command=self.update_proactive_setting)
        proactive_check.pack(padx=10, pady=5)
        
        # Interval setting
        interval_frame = ttk.Frame(proactive_frame)
        interval_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(interval_frame, text="Check-in interval (minutes):").pack(side='left')
        
        self.interval_var = tk.IntVar(value=self.auto_message_interval // 60)
        interval_spin = ttk.Spinbox(interval_frame, from_=1, to=60, 
                                   textvariable=self.interval_var,
                                   command=self.update_interval_setting,
                                   width=10)
        interval_spin.pack(side='right')
        
        # Save button
        ttk.Button(prefs_window, text="Save Preferences", 
                  style='Modern.TButton',
                  command=lambda: prefs_window.destroy()).pack(pady=20)
    
    def update_proactive_setting(self):
        """Update proactive messaging setting"""
        self.proactive_enabled = self.proactive_var.get()
        self.profile['proactive_enabled'] = self.proactive_enabled
        self.save_profile()
    
    def update_interval_setting(self):
        """Update check-in interval setting"""
        self.auto_message_interval = self.interval_var.get() * 60  # Convert to seconds
        self.profile['auto_message_interval'] = self.auto_message_interval
        self.save_profile()
    
    def show_model_settings(self):
        """Show model settings dialog"""
        model_window = tk.Toplevel(self.root)
        model_window.title("AI Model Settings")
        model_window.geometry("600x500")
        model_window.configure(bg=self.colors['bg_primary'])
        
        # Model status
        status_frame = ttk.LabelFrame(model_window, text=" Current Status ")
        status_frame.pack(fill='x', padx=10, pady=10)
        
        if hasattr(self, 'chloe') and hasattr(self.chloe, 'current_model') and self.chloe.current_model:
            status_text = f"✅ Currently using: {self.chloe.current_model}"
            status_color = self.colors['success']
        else:
            status_text = "❌ No model loaded"
            status_color = self.colors['error']
        
        status_label = tk.Label(status_frame, text=status_text,
                               bg=self.colors['bg_secondary'],
                               fg=status_color,
                               font=('Segoe UI', 12, 'bold'))
        status_label.pack(padx=10, pady=10)
        
        # Connection info
        info_text = """
        Chloe uses AI models through Ollama to provide intelligent responses.
        
        To use AI features:
        1. Install Ollama from https://ollama.com
        2. Run 'ollama serve' to start the service
        3. Download a compatible model (like llama2 or mistral)
        
        Without AI models, Chloe will use basic response patterns.
        """
        
        info_label = tk.Label(model_window, text=info_text,
                             bg=self.colors['bg_primary'],
                             fg=self.colors['text_secondary'],
                             font=('Segoe UI', 10),
                             justify='left')
        info_label.pack(padx=20, pady=10)
    
    def show_preferences_setup(self):
        """Show initial preferences setup during onboarding"""
        self.add_message("Chloe", 
                        "Perfect! Now let's set up your preferences.\n\n"
                        "I can send you gentle check-ins if we haven't chatted in a while. "
                        "Would you like me to do that? Just say 'yes' or 'no'.")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    pretty_print("🤖 Starting Chloe Mental Health Assistant...\n", "green")
    ollama_installed, ollama_running = check_ollama_full_status()
    ollama_ok = ollama_installed and ollama_running

    if not ollama_installed:
        print_ollama_help()
        sys.exit(0)

    elif not ollama_running:
        pretty_print("\nOllama is installed but NOT running.", "yellow")
        pretty_print("Start Ollama with:", "cyan", False)
        print("   ollama serve")
        pretty_print("Falling back to BASIC mode (no LLM features).\n", "yellow")
        sys.exit(0)

    # ————————————————————————
    # If we just reset, wipe the old DB and JSON right away
    if os.path.exists('reset_pending'):
        shutil.rmtree(CHROMADB_PATH, ignore_errors=True)
        for fn in ('chloe_profile.json', 'chloe_history.json', 'chloe_convo.json'):
            try: os.remove(fn)
            except: pass
        os.remove('reset_pending')
    # ————————————————————————

    if ollama_ok:
        ensure_model_ready("llama3.2")  # Blocks with loading UI

    # Only now create the main GUI
    try:
        app = ModernChloeGUI(ollama_ok=ollama_ok)
        app.run()
    except Exception as e:
        pretty_print(f"Error starting application: {e}", "red")
        try:
            messagebox.showerror("Startup Error", f"Failed to start Chloe: {e}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
