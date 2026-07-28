# Special K - Project Documentation

## 1. Project Overview
**Special K** is a local, privacy-focused **Parallel Intelligence Meeting Assistant**. It features a synchronized split-view interface with:
- **Fast Track (Left Pane)**: Live transcription using MLX-Whisper or WhisperX
- **Slow Track (Right Pane)**: Deep analysis extracting Topics, Decisions, and Action Items

**Core Philosophy:**
* **Local-First:** No audio or context data is sent to the cloud (unless the user configures a remote LLM endpoint).
* **Single Brain Architecture:** One shared AI model handles both live transcription and background analysis with priority queuing.
* **Click-to-Seek:** Every insight is linked to its source in the transcript for instant navigation.

---

## 2. Technical Architecture

### A. Frontend (UI)
* **Framework:** `Streamlit`
* **File:** `app.py`
* **Responsibilities:**
    * Manages the split-view UI (Left: Transcript, Right: Analysis)
    * Handles file uploads and displays the file debugger.
    * Sends chat requests to the LLM (Fast Track priority).
    * Refreshes the live transcript and analysis views using `@st.fragment`.

### B. Audio Engine (Backend)
* **File:** `audio_engine.py`
* **Responsibilities:**
    * **Recording:** Captures raw audio frames from the selected microphone device.
    * **VAD (Voice Activity Detection):** Filters silence to prevent hallucination.
    * **Transcription:**
        * *Apple Silicon:* Uses `mlx-whisper` for optimized performance.
        * *CUDA/CPU:* Uses `whisperx` or standard `whisper`.
    * **Threading:** Runs recording and transcription in separate background threads to keep the UI responsive.
    * **Live Correction:** Re-transcribes the growing audio buffer every 0.5 seconds, allowing Whisper to correct itself as more context becomes available.

### C. Analysis Engine (NEW)
* **File:** `analysis_engine.py`
* **Core Components:**
    * **SingleBrainController:** Priority-based queue managing model access
        - Priority 1 (Fast Track): Chat and live formatting - never lags
        - Priority 2 (Slow Track): Background analysis - runs when idle
    * **TranscriptManager:** Tracks segments with unique IDs for click-to-seek
    * **TopicLifecycleEngine:** Manages topic states (Active, Resolved, Parking Lot)
    * **InsightExtractor:** Extracts decisions, action items, key points
    * **RollingContextManager:** Maintains context between analysis cycles for deduplication

### D. Data Management
* **File:** `meeting_manager.py`
* **Storage:**
    * `meetings.json`: Stores metadata (ID, title, date, file paths).
    * `settings.json`: Stores user settings including LLM models and transcription preferences.
    * `context_data/`: Stores uploaded files (PDFs, DOCX, TXT) renamed with timestamps.
    * `*.txt`: Stores transcripts with embedded analysis data (JSON).
* **Responsibilities:**
    * CRUD operations for meetings and prompts.
    * Physical file saving and deletion.

### E. LLM Integration
* **Endpoint:** Configured to `http://localhost:1234/v1/chat/completions` (compatible with LM Studio, Ollama, etc.).
* **Context Construction:**
    * **System Prompt:** Injected with user-defined style instructions (Tone, Detail, Language).
    * **User Prompt:** Combined with `Transcript Text` + `File Contents`.

---

## 3. Parallel Intelligence Features

### A. Single Brain Architecture

**Priority Queue System:**
```
Priority 1 (Fast Track): User chat, live transcript queries
Priority 2 (Slow Track): Background analysis, topic extraction
```

The system ensures that:
- Fast Track tasks are processed immediately
- Slow Track tasks only run when the model is idle
- Slow Track can be interrupted if Fast Track needs the model

### B. Click-to-Seek Interface

**Data Lineage:**
- Every transcript segment has a unique ID (`seg_xxxx`)
- Insights store `source_segment_ids` linking to their origin
- Clicking 🔗 on any insight scrolls to the source segment

**UI Behavior:**
- Left Pane highlights the selected segment with a blue border
- Smooth transitions between selections

### C. Topic Lifecycle Engine

**States:**
1. **🟢 Active:** Currently being discussed
2. **✅ Resolved:** Topic concluded with or without a decision
3. **🅿️ Parking Lot:** Introduced but abandoned/not meaningfully discussed

**Auto-Categorization Rules:**
- Topics inactive for >3 minutes with ≤2 mentions → Parking Lot
- Parking Lot topics mentioned again → Back to Active
- Topics with explicit resolution → Resolved

### D. Rolling Context Awareness

**Context Passing:**
- Each analysis cycle receives a summary of previous analysis
- Recent topics, decisions, and actions are passed for deduplication

**Smart Deduplication:**
- Uses 80% word overlap threshold to detect duplicates
- Previous decisions/actions are explicitly listed in prompts

**Reference Resolution:**
- Context helps LLM understand vague references ("that", "it")
- Previous segment summary provides disambiguation context

---

## 4. Data Flow Diagram

1.  **Input:** User selects Mic & Uploads Files (PDF/DOCX).
2.  **Processing (Fast Track - Live):**
    * Audio -> `AudioRecorder` -> `Whisper Model` -> Text
    * Text -> `TranscriptManager` -> Segments with IDs
3.  **Processing (Slow Track - Background):**
    * Segments (buffered ~100 words) -> `AnalysisEngine`
    * Analysis -> `TopicLifecycleEngine` + `InsightExtractor`
    * Context updated in `RollingContextManager`
4.  **Interaction:**
    * User clicks insight -> Scrolls to source segment
    * User asks question -> Fast Track LLM call
5.  **Storage (On Stop):**
    * Transcript + Analysis JSON saved to disk
    * Context files moved to `context_data/`
    * Metadata updated in `meetings.json`

---

## 5. Output Format

Saved transcripts include both transcript and analysis data:

```
=== TRANSCRIPT ===
[00:00:15] First segment of speech...

[00:00:45] Second segment of speech...

=== ANALYSIS DATA ===
{
  "topics": {
    "active": [...],
    "resolved": [...],
    "parking_lot": [...]
  },
  "insights": {
    "decisions": [...],
    "actions": [...],
    "key_points": [...]
  }
}
```

---

## 6. Key Functions & Logic

### `analysis_engine.py`

* `SingleBrainController`:
    * `call_llm()`: Direct LLM call with configurable priority
    * `is_model_idle()`: Check if Slow Track can run
    * `should_interrupt_slow_track()`: Check for pending Fast Track tasks

* `TranscriptManager`:
    * `add_segment()`: Add segment with unique ID
    * `get_segment_by_id()`: For click-to-seek navigation
    * `get_segments_since()`: Get unanalyzed segments

* `TopicLifecycleEngine`:
    * `add_or_update_topic()`: Add new or update existing topic
    * `resolve_topic()`: Mark topic as resolved
    * `update_lifecycle()`: Auto-categorize based on time/mentions

* `RollingContextManager`:
    * `get_context_prompt()`: Format context for LLM injection
    * `update_context()`: Update after each analysis cycle

### `app.py`

* `show_live_transcript()`: Fragment that updates every 1s
* `show_analysis_insights()`: Fragment that updates every 3s
* Click handlers set `selected_segment_id` for visual highlighting

* `get_text_from_files(file_list)`:
    * **Crucial:** Handles extraction of text from binary formats.
    * **Supported:** `.pdf` (via `pypdf`), `.docx` (via `python-docx`), `.txt/.md/.py`.

* `LLMTranscriptionManager`:
    * `reset()`: Call when starting new recording session.
    * `update_and_format(raw_text)`: Non-blocking, triggers background formatting.
    * `get_formatted_output()`: Returns current formatted text immediately.

### `meeting_manager.py`

* `update_meeting_title(m_id, new_title)`: Updates meeting title in meetings.json.
* `load_settings()` / `save_settings()`: Manages settings.json including LLM models.

### `audio_engine.py`

* `_transcribe_loop()`:
    * Runs every 0.5 seconds while recording.
    * Re-transcribes the entire audio buffer for self-correction.
    * Updates `current_live_text` which may change until finalized.

* `_finalize_segment()`:
    * Called on silence or when buffer gets too long.
    * Commits `current_live_text` to `committed_text`.
    * Manages overlap for continuity between segments.

---

## 7. Settings Configuration

Located in `settings.json`:
- `llm_model_id`: Active LLM model for chat and analysis
- `saved_llm_models`: Array of saved model identifiers
- `transcription_engine`: "MLX Whisper (Apple Silicon)" or "WhisperX (Torch)"
- `transcription_model`: Whisper model path/name

---

## 8. Setup & Dependencies

To run this project, the environment must include:
* `streamlit`
* `pypdf`
* `python-docx`
* `requests`
* **Inference Engine:** `mlx-whisper` (Mac) OR `whisperx` (Linux/Windows)
* **LLM Server:** LM Studio, Ollama, or compatible OpenAI API endpoint

---

## 9. How to Work on This Project (LLM Instructions)

If you are an AI assistant helping to modify this code:

1. **Preserve Imports:** Do not remove the `try/except` blocks for `pypdf` and `docx` imports.

2. **Analysis Engine:**
    * The analysis runs in background threads - never block the main UI
    * Priority 1 (Fast Track) must never wait for Priority 2 (Slow Track)
    * All segments need unique IDs for click-to-seek

3. **Topic Lifecycle:**
    * Topics automatically move between states based on time and mentions
    * Parking Lot is for topics that were "dropped" - not for resolved topics

4. **Rolling Context:**
    * Always pass previous context to avoid duplicates
    * The LLM prompt explicitly lists recent decisions/actions to prevent repetition

5. **State Management:**
    * Use `st.session_state` for all UI state
    * `selected_segment_id` controls transcript highlighting

6. **Click-to-Seek:**
    * Every insight must have `source_segment_ids`
    * The 🔗 button sets `selected_segment_id` and triggers rerun

7. **Testing:**
    * Test with actual recordings to verify analysis quality
    * Ensure Fast Track never lags even during intensive Slow Track analysis

8. **Concurrency:**
    * The audio engine runs in threads
    * Do not attempt to run blocking audio code in the main Streamlit thread
    * Always run LLM calls in background threads for Slow Track
