# 🥣 Special K - Local-First Meeting Assistant

A **privacy-preserving** meeting intelligence tool for TPMs, EMs, SDEs that handle confidential information without exposing it to remote servers. Unlike cloud-based meeting assistants (like Granola.ai), Special K keeps all audio, transcripts, and analysis entirely on your machine using a local LLM.

Built with voice coding in Python/Streamlit during my Staff TPM role, where protecting confidential cross-functional data is non-negotiable.

---

## 🎯 Why Special K?

### The Problem
Traditional meeting assistants (Granola, Fireflies, Otter) require sending audio and transcripts to remote servers. For Staff TPMs handling confidential information—strategy, M&A discussions, salary reviews, customer negotiations—this creates unacceptable data exposure risks.

### The Solution
**Special K** runs entirely locally:
- 🎤 **Audio** stays on your machine
- 🧠 **Transcription** via local Whisper model (MLX or WhisperX)
- 💭 **Analysis** using local LLM (LM Studio, Ollama, or Claude API with end-to-end encryption)
- 📝 **Insights** never leave your disk

No audio files. No transcripts. No data sent to any remote service. *Maximum privacy. Maximum control.*

### The Name
Special K is my favorite cereal. Granola is my second. So this project—a local alternative to Granola.ai—had to be Special K. 🥣

---

## ✨ Features

### Parallel Intelligence Architecture
**Split-View Interface** with dual processing pipelines:

- **Fast Track (Left Pane)** — Live Transcription
  - Real-time speech-to-text with MLX Whisper or WhisperX
  - Every sentence gets a unique ID for tracking
  - LLM-powered cleaning (optional) for readable output
  - Click-to-seek integration with insights

- **Slow Track (Right Pane)** — Deep Analysis (runs in background)
  - **Topics** with lifecycle tracking (Active → Resolved → Parking Lot)
  - **Decisions** extracted from conversation
  - **Action Items** automatically identified
  - **Smart deduplication** using rolling context

### Single Brain Architecture
One shared AI model handles both pipelines via priority queuing:
- **Priority 1 (Fast Track):** Chat & live transcription—never lags
- **Priority 2 (Slow Track):** Background analysis—only runs when idle

### Context-Aware Intelligence
- Upload PDFs, Word docs, or text files as context
- LLM uses transcript + context to answer questions
- Click any insight to jump to its source in the transcript
- Chat history persists per meeting

---

## 🚀 Quick Start

### Prerequisites
1. **Python 3.10+**
2. **Local LLM Server** (pick one):
   - [LM Studio](https://lmstudio.ai/) (easiest, supports most models)
   - [Ollama](https://ollama.ai/)
   - Any OpenAI-compatible endpoint
3. **Microphone** on your machine

### Installation

```bash
git clone https://github.com/yourusername/specialk.git
cd specialk
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

### Running the App

1. **Start your LLM server** (default: LM Studio on `http://localhost:1234`)
2. **Run Special K:**
   ```bash
   streamlit run app.py
   ```
3. Click **+ New Meeting** and start recording

---

## ⚙️ Configuration

### LLM Provider
Choose between:
- **Local LLM** (default): Point to your local LM Studio, Ollama, or equivalent
- **Claude API**: Use Anthropic's Claude with your API key (still process audio locally)

### Audio Transcription
- **Apple Silicon Mac:** MLX Whisper (optimized, fast)
- **Linux/Windows/CUDA:** WhisperX (more accurate with speaker diarization)

### Optional: LLM-Powered Transcription Cleaning
Enable in settings to have the LLM automatically clean and format raw transcripts for readability. Disabled by default (raw transcription is usually good enough).

---

## 📁 Project Structure

```
specialk/
├── app.py                    # Main Streamlit UI
├── audio_engine.py           # Recording + local transcription
├── analysis_engine.py        # Topic & insight extraction
├── meeting_manager.py        # Data storage (meetings.json, etc.)
├── llm_provider.py           # LLM abstraction (local + Claude)
├── prompts.json              # Custom prompts library
├── settings.json             # User configuration
├── meetings.json             # Meeting metadata
└── meetings/                 # Meeting transcripts & analysis
    └── {timestamp}/
        └── transcript.txt    # Saved meeting data
```

---

## 🔐 Privacy by Design

- **No Cloud Uploads:** All audio stays local during transcription
- **No Telemetry:** Zero analytics, tracking, or remote logging
- **No Account Required:** Runs completely standalone
- **Full Data Ownership:** All recordings and analysis are yours to keep, delete, or export
- **Optional Remote LLM:** If using Claude API, audio is still processed locally—only final LLM requests go remote (with your own API key)

---

## 🧠 How It Works

### During Recording (Real-Time)
1. **Audio Input** → Whisper model transcribes locally
2. **Fast Track** displays live transcript (with optional LLM cleaning)
3. **Slow Track** runs analysis in background:
   - Detects topics, decisions, action items
   - Maintains rolling context to avoid duplicates
   - Auto-categorizes topic status

### After Recording
- View cleaned transcript, raw transcript, and extracted analysis
- Ask questions about the meeting using chat interface
- Upload additional context files for reference
- Export or delete meetings

---

## 🛠️ Built With

- **UI Framework:** [Streamlit](https://streamlit.io/)
- **Local Transcription:** [MLX Whisper](https://github.com/ml-explore/mlx-whisper) (Mac) or [WhisperX](https://github.com/m-bain/whisperx) (Linux/Windows)
- **LLM Integration:** OpenAI-compatible APIs (LM Studio, Ollama) + Anthropic Claude SDK
- **Backend:** Python with threading for responsive UI
- **Storage:** JSON-based file storage (no database required)

---

## 📖 For Staff TPMs & Confidentiality-Conscious Teams

Special K was built specifically for roles that handle sensitive information:
- **Executive Strategy Meetings** – Keep M&A discussions private
- **Compensation Reviews** – Salary and promotion data stays encrypted locally
- **Customer Negotiations** – Pricing and contract details don't touch cloud
- **Cross-Functional Alignment** – Candid conversations without data residency concerns
- **Compliance-Critical Discussions** – HIPAA, SOX, GDPR compliant by design

---

## 🔧 Development

### Adding Custom Prompts
Use the **⚡ Custom Prompts** section in the sidebar to create analysis templates.

### Extending Analysis
Edit `analysis_engine.py` to:
- Add new insight types (sentiments, risks, etc.)
- Modify topic lifecycle rules
- Customize deduplication thresholds

### Using Different Models
Switch models in **⚙️ System Settings**:
- Add new local model IDs (from LM Studio, Ollama)
- Change transcription model
- Toggle Claude API vs. local LLM

---

## ⚠️ Known Limitations

- **Transcription Quality** depends on audio quality and background noise
- **Analysis relies on LLM capability** – smaller models may miss nuanced topics
- **Speaker Diarization** only with WhisperX (not MLX Whisper)
- **No realtime collaboration** – single-user per instance
- **Performance** dependent on local hardware

---

## 📝 License

[Add your license here—MIT, Apache 2.0, etc.]

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Performance optimizations for larger meetings
- Better topic deduplication algorithms
- Support for additional transcription engines
- UI/UX improvements

---

## 💬 Questions or Issues?

Open an issue on GitHub or reach out. This tool was built for serious work—your feedback matters.

---

**Built with ❤️ for confidential conversations.**
