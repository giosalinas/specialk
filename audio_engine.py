import threading
import queue
import time
import numpy as np
import pyaudio
import mlx_whisper

# --- MODEL CATALOG ---
MLX_MODELS = {
    "Large v3 Turbo (Recommended)": "mlx-community/whisper-large-v3-turbo",
    "Large v3": "mlx-community/whisper-large-v3",
    "Distil Large v3": "mlx-community/whisper-distil-large-v3",
    "Medium": "mlx-community/whisper-medium",
    "Base": "mlx-community/whisper-base",
    "Small": "mlx-community/whisper-small",
    "Tiny": "mlx-community/whisper-tiny",
}


class AudioRecorder:
    def __init__(self, settings):
        self.running = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.silence_threshold = 0.01

        # MLX Whisper model - validate the model name is a valid MLX model
        default_model = "mlx-community/whisper-large-v3-turbo"
        model_from_settings = settings.get("transcription_model", default_model)

        # Validate model is an MLX model (must start with mlx-community/)
        valid_models = set(MLX_MODELS.values())
        if model_from_settings in valid_models:
            self.model_name = model_from_settings
        elif model_from_settings.startswith("mlx-community/"):
            # Allow any mlx-community model even if not in catalog
            self.model_name = model_from_settings
        else:
            # Invalid model (likely old WhisperX model) - use default
            print(f"⚠️ Invalid model '{model_from_settings}', using default: {default_model}")
            self.model_name = default_model

        # Continuous transcript with inline timestamps
        self.committed_text = ""  # Finalized text (won't change)
        self.current_live_text = ""  # Currently being transcribed (may change)
        self.current_live_timestamp = None
        self.is_speaking = False

        # Overlap handling for continuity
        self.overlap_duration = 10  # seconds of audio to keep for context
        self.last_committed_words = []  # Last N words for deduplication

        # Thread safety
        self.buffer_lock = threading.Lock()
        self.transcript_lock = threading.Lock()

        # Tracking
        self.silence_count = 0
        self.max_silence_before_finalize = 4  # ~2 seconds of silence to finalize
        self.max_segment_duration = 45  # seconds before auto-checkpoint

        self.audio_buffer = np.array([], dtype=np.float32)
        self.load_model()

    def load_model(self):
        """Loads the MLX Whisper model."""
        print(f"🚀 Loading MLX Whisper | Model: {self.model_name}")

        try:
            # Warmup the model
            warmup = np.zeros(16000, dtype=np.float32)
            mlx_whisper.transcribe(warmup, path_or_hf_repo=self.model_name)
            print("✅ MLX Whisper Model Ready.")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()

    def get_devices(self):
        """Returns list of devices and the smart default index."""
        p = pyaudio.PyAudio()
        devices = []
        target_index = -1

        count = p.get_device_count()
        for i in range(count):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                name = info.get('name')
                devices.append((i, name))
                if "BlackInput" in name or "BlackHole" in name:
                    target_index = i
        p.terminate()

        if target_index == -1 and devices:
            target_index = devices[-1][0]

        return devices, target_index

    def start(self, device_index):
        self.running = True
        with self.transcript_lock:
            self.committed_text = ""
            self.current_live_text = ""
            self.current_live_timestamp = None
            self.is_speaking = False
            self.last_committed_words = []
        with self.buffer_lock:
            self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_queue = queue.Queue()
        self.silence_count = 0

        self.t_process = threading.Thread(target=self._process_loop, daemon=True)
        self.t_transcribe = threading.Thread(target=self._transcribe_loop, daemon=True)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            stream_callback=self._callback
        )

        self.t_process.start()
        self.t_transcribe.start()
        self.stream.start_stream()

    def stop(self):
        self.running = False
        # Finalize any remaining live text
        self._finalize_segment()

        try:
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
        except:
            pass

        try:
            if hasattr(self, 'p'):
                self.p.terminate()
        except:
            pass

    def _callback(self, in_data, frame_count, time_info, status):
        if self.running:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def _process_loop(self):
        while self.running:
            while not self.audio_queue.empty():
                data = self.audio_queue.get()
                with self.buffer_lock:
                    self.audio_buffer = np.concatenate((self.audio_buffer, data))

            # Auto-checkpoint before buffer gets too long (during continuous speech)
            with self.buffer_lock:
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                should_checkpoint = buffer_duration >= self.max_segment_duration and self.is_speaking

            if should_checkpoint:
                self._finalize_segment(keep_overlap=True)

            time.sleep(0.05)

    def _transcribe_loop(self):
        """
        Continuous transcription with inline timestamps:
        1. During speech: Show constantly updated transcription
        2. On checkpoint: Commit text with timestamp, keep overlap for context
        3. On silence: Finalize and clear
        """
        while self.running:
            time.sleep(0.5)

            if not self.running:
                break

            with self.buffer_lock:
                # Need at least 1 second of audio
                if len(self.audio_buffer) < self.sample_rate:
                    continue
                audio_copy = self.audio_buffer.copy()

            # Check volume in recent audio
            recent_samples = min(int(0.3 * self.sample_rate), len(audio_copy))
            recent_audio = audio_copy[-recent_samples:]
            volume = np.sqrt(np.mean(recent_audio**2))

            if volume > self.silence_threshold:
                self.silence_count = 0

                if not self.is_speaking:
                    self.is_speaking = True
                    with self.transcript_lock:
                        self.current_live_timestamp = time.strftime("%H:%M:%S")

                try:
                    result = mlx_whisper.transcribe(
                        audio_copy,
                        path_or_hf_repo=self.model_name,
                        language="en"
                    )
                    full_text = result.get('text', '').strip()

                    if full_text:
                        # Remove overlap from committed text
                        display_text = self._remove_overlap(full_text)
                        with self.transcript_lock:
                            self.current_live_text = display_text

                except Exception as e:
                    print(f"Transcription Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                self.silence_count += 1

                # After silence, finalize completely
                if self.silence_count >= self.max_silence_before_finalize and self.is_speaking:
                    self._finalize_segment(keep_overlap=False)

    def _remove_overlap(self, text):
        """
        Remove overlapping text that was already committed.
        Uses minimum match length of 5 to prevent deleting common phrases.
        """
        if not self.last_committed_words or not text:
            return text

        text_words = text.split()
        if not text_words:
            return text

        best_match_end = 0
        MIN_MATCH_LEN = 5

        # Try different match lengths, starting from longer sequences
        for match_len in range(min(25, len(self.last_committed_words)), MIN_MATCH_LEN - 1, -1):
            search_words = self.last_committed_words[-match_len:]
            search_phrase = ' '.join(w.lower().strip('.,!?;:') for w in search_words)

            # Look for this phrase in the first part of new text
            for i in range(min(20, len(text_words) - match_len + 1)):
                candidate_words = text_words[i:i+match_len]
                candidate = ' '.join(w.lower().strip('.,!?;:') for w in candidate_words)

                if candidate == search_phrase:
                    match_end = i + match_len
                    if match_end > best_match_end:
                        best_match_end = match_end
                        if match_len >= 8:  # Early exit on strong match
                            break
            if best_match_end > 0 and match_len >= 8:
                break

        if best_match_end > 0:
            new_text = ' '.join(text_words[best_match_end:])
            return new_text.strip()

        return text

    def _find_sentence_boundary(self, text):
        """
        Find a good sentence boundary to split text for checkpointing.
        Returns (text_to_commit, text_to_keep) tuple.
        """
        if not text:
            return (None, None)

        # Look for sentence-ending punctuation
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

        best_split = -1
        for ending in sentence_endings:
            idx = text.rfind(ending)
            if idx > best_split:
                best_split = idx + len(ending) - 1  # Include the punctuation

        if best_split > 0 and best_split < len(text) - 1:
            commit_text = text[:best_split + 1].strip()
            keep_text = text[best_split + 1:].strip()

            # Only split if we have meaningful content on both sides
            if len(commit_text) > 10 and len(keep_text) > 5:
                return (commit_text, keep_text)

        # No good boundary found
        return (text, None)

    def _finalize_segment(self, keep_overlap=False):
        """Commit current live text to the transcript."""
        text_to_commit = None
        text_to_keep = None

        with self.transcript_lock:
            if self.current_live_text:
                timestamp = self.current_live_timestamp or time.strftime("%H:%M:%S")

                # If keeping overlap, try to find a good sentence boundary
                if keep_overlap:
                    commit_text, keep_text = self._find_sentence_boundary(self.current_live_text)

                    # Safety valve: if kept text is too long, force full commit
                    if keep_text and len(keep_text.split()) > 15:
                        commit_text = self.current_live_text
                        keep_text = None

                    if commit_text:
                        text_to_commit = commit_text
                        text_to_keep = keep_text
                    else:
                        text_to_commit = self.current_live_text
                        text_to_keep = None
                else:
                    text_to_commit = self.current_live_text
                    text_to_keep = None

                if text_to_commit:
                    # Add to committed text with timestamp on new line
                    if self.committed_text:
                        self.committed_text += f"\n[{timestamp}] {text_to_commit}"
                    else:
                        self.committed_text = f"[{timestamp}] {text_to_commit}"

                    # Save last words for overlap detection
                    all_words = self.committed_text.split()
                    self.last_committed_words = all_words[-50:] if len(all_words) > 50 else all_words

                # Update live text
                if text_to_keep:
                    self.current_live_text = text_to_keep
                else:
                    self.current_live_text = ""
                    if not keep_overlap:
                        self.current_live_timestamp = None
                        self.is_speaking = False

        # Handle audio buffer
        with self.buffer_lock:
            if keep_overlap and len(self.audio_buffer) > 0:
                overlap_samples = self.overlap_duration * self.sample_rate
                if len(self.audio_buffer) > overlap_samples:
                    self.audio_buffer = self.audio_buffer[-overlap_samples:]
            else:
                self.audio_buffer = np.array([], dtype=np.float32)

        self.silence_count = 0

    def get_transcript_text(self):
        """Returns continuous transcript with inline timestamps."""
        with self.transcript_lock:
            if self.current_live_text:
                ts = self.current_live_timestamp or time.strftime("%H:%M:%S")
                if self.committed_text:
                    return f"{self.committed_text} [{ts}] {self.current_live_text} ▌"
                else:
                    return f"[{ts}] {self.current_live_text} ▌"
            else:
                return self.committed_text

    # For backwards compatibility
    @property
    def transcript(self):
        with self.transcript_lock:
            return [self.committed_text] if self.committed_text else []

    @property
    def committed_transcript(self):
        return self.transcript

    @property
    def finalized_segments(self):
        with self.transcript_lock:
            if self.committed_text:
                return [{"timestamp": "", "text": self.committed_text}]
            return []
