import json
import os
from datetime import datetime

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEETINGS_FILE = os.path.join(BASE_DIR, "meetings.json")
PROMPTS_FILE = os.path.join(BASE_DIR, "prompts.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
CONTEXT_DIR = os.path.join(BASE_DIR, "context_data")

# --- HELPER: ROBUST JSON IO ---
def _load_json(filepath, default_value):
    if not os.path.exists(filepath): return default_value
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else default_value
    except: return default_value

def _save_json(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)
    except: pass

# --- HELPER: GENERIC FILE SAVER ---
def _save_file_to_disk(uploaded_file):
    """Saves any uploaded file to disk with a timestamp prefix."""
    if not os.path.exists(CONTEXT_DIR): os.makedirs(CONTEXT_DIR, exist_ok=True)

    timestamp = int(datetime.now().timestamp())
    # NO FILTER: We save everything (pdf, docx, txt, etc.)
    safe_name = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(CONTEXT_DIR, safe_name)

    try:
        uploaded_file.seek(0)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
    except Exception as e:
        print(f"Error saving {uploaded_file.name}: {e}")
        return None

# --- SETTINGS & PROMPTS ---
def load_settings():
    return _load_json(SETTINGS_FILE, {
        "llm_model_id": "local-model",
        "saved_llm_models": ["local-model"],
        "llm_provider": "local",
        "claude_api_key": "",
        "claude_model_id": "claude-sonnet-4-20250514",
        "llm_local_url": "http://localhost:1234/v1/chat/completions",
        "llm_transcription_enabled": False,
        "llm_transcription_model_id": "local-model",
        "llm_transcription_provider": "local",
        "claude_transcription_model_id": "claude-sonnet-4-20250514"
    })

def save_settings(s): _save_json(SETTINGS_FILE, s)

def load_prompts():
    return _load_json(PROMPTS_FILE, [{"title": "Summarize", "text": "Summarize this.", "icon": "📝", "usage_count": 0}])

def add_prompt(title, text, icon):
    p = load_prompts(); p.append({"title": title, "text": text, "icon": icon, "usage_count": 0}); _save_json(PROMPTS_FILE, p)

def delete_prompt(idx):
    p = load_prompts(); p.pop(idx); _save_json(PROMPTS_FILE, p)

def update_prompt(idx, title, text, icon):
    p = load_prompts()
    usage = p[idx].get("usage_count", 0)
    p[idx] = {"title": title, "text": text, "icon": icon, "usage_count": usage}
    _save_json(PROMPTS_FILE, p)

def increment_prompt_usage(prompt_text):
    """Increment the usage count for a prompt by matching its text."""
    p = load_prompts()
    for prompt in p:
        if prompt['text'].strip() == prompt_text.strip():
            prompt['usage_count'] = prompt.get('usage_count', 0) + 1
            _save_json(PROMPTS_FILE, p)
            return
    _save_json(PROMPTS_FILE, p)

def get_top_prompts(limit=10):
    """Get the top N most used prompts, sorted by usage count."""
    p = load_prompts()
    # Ensure all prompts have usage_count
    for prompt in p:
        if 'usage_count' not in prompt:
            prompt['usage_count'] = 0
    # Sort by usage count descending, then return top N
    sorted_prompts = sorted(p, key=lambda x: x.get('usage_count', 0), reverse=True)
    return sorted_prompts[:limit]

# --- MEETINGS ---
def load_meetings():
    data = _load_json(MEETINGS_FILE, [])
    return sorted(data, key=lambda x: x.get('id', 0), reverse=True) if isinstance(data, list) else []

def save_meeting(title, transcript_text, context_files, chat_history):
    meetings = load_meetings()
    timestamp = int(datetime.now().timestamp())

    # 1. Save Transcript
    t_path = os.path.join(BASE_DIR, f"transcript_{timestamp}.txt")
    with open(t_path, "w", encoding="utf-8") as f: f.write(transcript_text)

    # 2. Save Context Files
    saved_paths = []
    if context_files:
        for uf in context_files:
            path = _save_file_to_disk(uf)
            if path: saved_paths.append(path)

    new_m = {
        "id": timestamp,
        "title": title or "Untitled",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "transcript_path": t_path,
        "context_files": saved_paths,
        "chat_history": chat_history
    }
    meetings.insert(0, new_m); _save_json(MEETINGS_FILE, meetings)
    return new_m

def add_files_to_meeting(meeting_id, new_files):
    """Adds files to an existing meeting (Offline Mode)."""
    meetings = load_meetings()
    saved_paths = []

    # Save files to disk
    for uf in new_files:
        path = _save_file_to_disk(uf)
        if path: saved_paths.append(path)

    # Update JSON
    if saved_paths:
        for m in meetings:
            if m['id'] == meeting_id:
                m.setdefault('context_files', []).extend(saved_paths)
                _save_json(MEETINGS_FILE, meetings)
                return saved_paths
    return []

def update_meeting_chat(m_id, history):
    ms = load_meetings()
    for m in ms:
        if m['id'] == m_id: m['chat_history'] = history; break
    _save_json(MEETINGS_FILE, ms)

def update_meeting_title(m_id, new_title):
    ms = load_meetings()
    for m in ms:
        if m['id'] == m_id: m['title'] = new_title; break
    _save_json(MEETINGS_FILE, ms)

def delete_meeting(m_id):
    ms = load_meetings()
    target = next((m for m in ms if m['id'] == m_id), None)
    if target:
        if os.path.exists(target.get('transcript_path', '')):
            try: os.remove(target['transcript_path'])
            except: pass
        for fp in target.get('context_files', []):
            if os.path.exists(fp):
                try: os.remove(fp)
                except: pass
        ms = [m for m in ms if m['id'] != m_id]; _save_json(MEETINGS_FILE, ms)
