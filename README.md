# C.H.L.O.E.-V1

# Chloe Mental Health Assistant

Chloe is a modern, friendly AI-powered mental health companion with a beautiful desktop interface. Built on Python and Ollama LLMs, Chloe offers supportive conversation, mood analysis, and privacy-focused, offline-first architecture.
**Perfect for anyone who wants a supportive, smart, and caring chat companion that respects your privacy.**

---

## 🚀 Features

* **Conversational Mental Health Support:** Empathetic, AI-driven dialogue using local LLMs (via [Ollama](https://ollama.com/)).
* **Modern Desktop GUI:** Polished, animated chat interface with dark mode.
* **Mood Tracking & Insights:** Visualize, analyze, and sort your conversation history by mood, happiness, and more.
* **Privacy First:** All your data is stored locally; nothing is ever uploaded to the cloud.
* **Offline/Online Modes:** Chloe gracefully falls back to basic support if AI models aren’t available.
* **Import/Export Data:** Manage your chat history and profile easily.
* **Proactive Check-Ins:** Gentle, customizable reminders if you’re idle for a while.
* **Gen-Z & Adult Modes:** Chloe adapts her tone and vocabulary based on your age.

---

## 🛠️ Installation & Setup

### 1. **Clone the repository**

```bash
git clone [https://github.com/yourusername/chloe-ai.git](https://github.com/Malaz1512/C.H.L.O.E.-V1.git)
cd C.H.L.O.E.-V1
pip install -r requirements.txt
python chloe.py
```

### 2. **Install Python dependencies**

It’s recommended to use Python 3.8–3.13.
Create a venv if you like:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

*You may need to install `tkinter` manually on some Linux distros:*

```bash
sudo apt-get install python3-tk
```

---

### 3. **Install and set up Ollama (for local AI models)**

Chloe uses [Ollama](https://ollama.com/) for fast, private, and free LLM chat.

1. **Download and install Ollama:**
   [https://ollama.com/download](https://ollama.com/download)

2. **Start the Ollama server:**
   Open a terminal and run:

   ```bash
   ollama serve
   ```

3. **(First Run) Chloe will automatically download the default model (`llama3.2`) when needed.**

---

### 4. **Run Chloe!**

```bash
python chloe.py
```

That’s it! On first launch, Chloe will onboard you (name, age, preferences) and guide you from there.

---

## ⚙️ Usage

* **Chat naturally:** Type and send messages; Chloe will respond empathetically and intelligently.
* **Switch AI models:** Use the Model tab in the menu bar to load other models (advanced/experimental).
* **View mood analytics:** See your mood and happiness trends in the Analysis menu.
* **Export/Import data:** Use the File menu to back up or restore your conversations.

---

## 🧠 Troubleshooting

**Ollama Not Found?**

* Make sure you’ve installed [Ollama](https://ollama.com/download) and run `ollama serve` before starting Chloe.

**Model Not Downloading?**

* Ensure your internet connection is active on first run (for downloading models).
* On Linux: If you get permission issues, try running with `sudo` or fixing permissions for `.ollama`.

**No Response from Chloe?**

* Make sure the Ollama server is running (`ollama serve`).
* If still stuck, restart both Ollama and Chloe, and check for errors in the terminal.
* If stuck after both steps, maybe LLM model is too large for your computer, switch to the small models and see if it is still stays stuck.

---

## 🛡️ Privacy & Data

* **All user data (profile, conversation, mood history) is stored locally in `.json` files.**
* No internet connection is required after the models are downloaded.
* Chloe never sends your data to the cloud or to any third party.

---

## 📝 Customization & Contributing

* Fork, star, or submit pull requests!
* The UI, onboarding, and AI prompt logic are all hackable—tweak Chloe to your liking.
* All feedback welcome.

---

## 🪴 Credits

* Built with Python, Tkinter, and [Ollama](https://ollama.com/).
* Inspired by a vision of accessible, nonjudgmental mental health support for all.

---

## 📄 License

MIT License.
See [LICENSE](LICENSE) for full details.

---

### Questions or Suggestions?

Open an issue or start a discussion on [GitHub](https://github.com/yourusername/chloe-ai/issues).
Chloe loves feedback! 💚

---

**Made with care by Malaz Nakaweh and Husnain Sidhu**
