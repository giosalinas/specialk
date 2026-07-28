"""
Analysis Engine - The "Single Brain" Architecture

This module implements a priority-based analysis system that shares a single AI model
between live transcription (Fast Track) and deep analysis (Slow Track).

Priority 1 (Fast Track): Live transcript - never lags
Priority 2 (Slow Track): Background analysis - runs only when model is idle

Key Features:
- Segment-based analysis with unique IDs for click-to-seek
- Topic lifecycle tracking (Active, Resolved, Parking Lot)
- Rolling context for deduplication and reference resolution
"""

import threading
import queue
import time
import uuid
import re
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from llm_provider import LLMProvider, LocalLLMProvider, get_provider


class TopicState(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    PARKING_LOT = "parking_lot"


class InsightType(Enum):
    TOPIC = "topic"
    DECISION = "decision"
    ACTION_ITEM = "action"
    KEY_POINT = "key_point"


@dataclass
class TranscriptSegment:
    """A segment of transcript text with a unique ID for tracking."""
    id: str
    text: str
    timestamp: str
    start_time: float
    word_count: int

    def to_dict(self):
        return asdict(self)


@dataclass
class Topic:
    """Tracks a conversation topic through its lifecycle."""
    id: str
    name: str
    state: TopicState
    source_segment_ids: List[str]
    first_mentioned: float
    last_mentioned: float
    mention_count: int = 1
    resolution_summary: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        d['state'] = self.state.value
        return d


@dataclass
class Insight:
    """An extracted insight (decision, action item, key point) linked to source segments."""
    id: str
    type: InsightType
    content: str
    source_segment_ids: List[str]
    timestamp: str
    topic_id: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        d['type'] = self.type.value
        return d


@dataclass
class AnalysisContext:
    """Rolling context passed between analysis cycles for continuity."""
    summary: str
    recent_topics: List[str]
    recent_decisions: List[str]
    recent_actions: List[str]
    last_segment_id: str
    total_segments_analyzed: int


class PriorityQueue:
    """Thread-safe priority queue for managing model access."""

    def __init__(self):
        self.lock = threading.Lock()
        self.high_priority_tasks = queue.Queue()  # Fast Track (chat, live formatting)
        self.low_priority_tasks = queue.Queue()   # Slow Track (analysis)
        self.is_processing = False

    def add_high_priority(self, task_fn, *args, **kwargs):
        """Add a high-priority task (Fast Track)."""
        self.high_priority_tasks.put((task_fn, args, kwargs))

    def add_low_priority(self, task_fn, *args, **kwargs):
        """Add a low-priority task (Slow Track)."""
        self.low_priority_tasks.put((task_fn, args, kwargs))

    def get_next_task(self):
        """Get next task, prioritizing high priority."""
        if not self.high_priority_tasks.empty():
            return self.high_priority_tasks.get(), 'high'
        elif not self.low_priority_tasks.empty():
            return self.low_priority_tasks.get(), 'low'
        return None, None

    def has_high_priority(self):
        """Check if there are high-priority tasks waiting."""
        return not self.high_priority_tasks.empty()


class SingleBrainController:
    """
    Central controller that manages the single AI model for both tracks.
    Ensures Fast Track (live) always takes priority over Slow Track (analysis).
    """

    def __init__(self, llm_url: str, model_id: str, provider: LLMProvider = None):
        self.llm_url = llm_url
        self.model_id = model_id
        self.provider = provider or LocalLLMProvider(base_url=llm_url, model_id=model_id)
        self.priority_queue = PriorityQueue()
        self.lock = threading.Lock()
        self.is_busy = False
        self.current_priority = None

        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the worker thread."""
        # Avoid starting if already running
        if self._running:
            return
        # Create a new thread each time (threads can only be started once)
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        self._worker_thread = None

    def _process_queue(self):
        """Worker thread that processes tasks in priority order."""
        while self._running:
            task_data, priority = self.priority_queue.get_next_task()

            if task_data:
                task_fn, args, kwargs = task_data
                with self.lock:
                    self.is_busy = True
                    self.current_priority = priority

                try:
                    task_fn(*args, **kwargs)
                except Exception as e:
                    print(f"Task error: {e}")
                finally:
                    with self.lock:
                        self.is_busy = False
                        self.current_priority = None
            else:
                time.sleep(0.1)

    def is_model_idle(self) -> bool:
        """Check if the model is available for low-priority work."""
        with self.lock:
            return not self.is_busy

    def should_interrupt_slow_track(self) -> bool:
        """Check if Slow Track should pause for Fast Track."""
        return self.priority_queue.has_high_priority()

    def call_llm(self, messages: List[Dict], temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
        """Make a direct LLM call using the configured provider."""
        return self.provider.chat_with_messages(messages, temperature, timeout)


class TranscriptManager:
    """
    Manages transcript segments with unique IDs for click-to-seek functionality.
    Every piece of text gets a permanent, traceable ID.
    """

    def __init__(self):
        self.segments: Dict[str, TranscriptSegment] = {}
        self.segment_order: List[str] = []
        self.lock = threading.Lock()
        self.recording_start_time: Optional[float] = None

    def reset(self):
        """Clear all segments for a new session."""
        with self.lock:
            self.segments.clear()
            self.segment_order.clear()
            self.recording_start_time = time.time()

    def add_segment(self, text: str, timestamp: str = None) -> TranscriptSegment:
        """Add a new segment with a unique ID."""
        with self.lock:
            if self.recording_start_time is None:
                self.recording_start_time = time.time()

            segment_id = f"seg_{uuid.uuid4().hex[:8]}"
            elapsed = time.time() - self.recording_start_time

            if timestamp is None:
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            segment = TranscriptSegment(
                id=segment_id,
                text=text.strip(),
                timestamp=timestamp,
                start_time=elapsed,
                word_count=len(text.split())
            )

            self.segments[segment_id] = segment
            self.segment_order.append(segment_id)

            return segment

    def get_segment(self, segment_id: str) -> Optional[TranscriptSegment]:
        """Get a segment by ID."""
        with self.lock:
            return self.segments.get(segment_id)

    def get_all_segments(self) -> List[TranscriptSegment]:
        """Get all segments in order."""
        with self.lock:
            return [self.segments[sid] for sid in self.segment_order]

    def get_segments_since(self, segment_id: str) -> List[TranscriptSegment]:
        """Get all segments after a given segment ID."""
        with self.lock:
            if segment_id not in self.segments:
                return list(self.segments.values())

            try:
                idx = self.segment_order.index(segment_id)
                return [self.segments[sid] for sid in self.segment_order[idx + 1:]]
            except ValueError:
                return []

    def get_full_transcript(self) -> str:
        """Get the full transcript text with timestamps."""
        with self.lock:
            parts = []
            for sid in self.segment_order:
                seg = self.segments[sid]
                parts.append(f"[{seg.timestamp}] {seg.text}")
            return "\n\n".join(parts)

    def get_segment_text_by_ids(self, segment_ids: List[str]) -> str:
        """Get combined text for given segment IDs."""
        with self.lock:
            texts = []
            for sid in segment_ids:
                if sid in self.segments:
                    texts.append(self.segments[sid].text)
            return " ".join(texts)


class TopicLifecycleEngine:
    """
    Tracks conversation topics through their lifecycle:
    - Active: Currently being discussed
    - Resolved: Concluded with or without a decision
    - Parking Lot: Introduced but abandoned/not meaningfully discussed
    """

    def __init__(self):
        self.topics: Dict[str, Topic] = {}
        self.lock = threading.Lock()

        self.inactive_threshold = 180
        self.parking_lot_threshold = 60

    def reset(self):
        """Clear all topics for a new session."""
        with self.lock:
            self.topics.clear()

    def add_or_update_topic(self, name: str, segment_id: str, current_time: float) -> Topic:
        """Add a new topic or update an existing one."""
        with self.lock:
            normalized_name = name.lower().strip()

            for topic_id, topic in self.topics.items():
                if topic.name.lower() == normalized_name:
                    topic.last_mentioned = current_time
                    topic.mention_count += 1
                    if segment_id not in topic.source_segment_ids:
                        topic.source_segment_ids.append(segment_id)

                    if topic.state == TopicState.PARKING_LOT:
                        topic.state = TopicState.ACTIVE

                    return topic

            topic_id = f"topic_{uuid.uuid4().hex[:8]}"
            topic = Topic(
                id=topic_id,
                name=name,
                state=TopicState.ACTIVE,
                source_segment_ids=[segment_id],
                first_mentioned=current_time,
                last_mentioned=current_time
            )
            self.topics[topic_id] = topic
            return topic

    def resolve_topic(self, topic_id: str, resolution_summary: str = None):
        """Mark a topic as resolved."""
        with self.lock:
            if topic_id in self.topics:
                self.topics[topic_id].state = TopicState.RESOLVED
                self.topics[topic_id].resolution_summary = resolution_summary

    def update_lifecycle(self, current_time: float):
        """Update topic states based on time elapsed since last mention."""
        with self.lock:
            for topic in self.topics.values():
                if topic.state == TopicState.ACTIVE:
                    time_since_mention = current_time - topic.last_mentioned

                    if time_since_mention > self.inactive_threshold:
                        if topic.mention_count <= 2:
                            topic.state = TopicState.PARKING_LOT

    def get_topics_by_state(self, state: TopicState) -> List[Topic]:
        """Get all topics in a given state."""
        with self.lock:
            return [t for t in self.topics.values() if t.state == state]

    def get_all_topics(self) -> Dict[str, List[Topic]]:
        """Get all topics grouped by state."""
        return {
            "active": self.get_topics_by_state(TopicState.ACTIVE),
            "resolved": self.get_topics_by_state(TopicState.RESOLVED),
            "parking_lot": self.get_topics_by_state(TopicState.PARKING_LOT)
        }


class InsightExtractor:
    """
    Extracts insights (decisions, action items, key points) from transcript segments.
    Links each insight to its source segment IDs for click-to-seek.
    """

    def __init__(self):
        self.insights: Dict[str, Insight] = {}
        self.lock = threading.Lock()

    def reset(self):
        """Clear all insights for a new session."""
        with self.lock:
            self.insights.clear()

    def add_insight(self, insight_type: InsightType, content: str,
                   source_segment_ids: List[str], timestamp: str,
                   topic_id: str = None) -> Insight:
        """Add a new insight linked to source segments."""
        with self.lock:
            insight_id = f"insight_{uuid.uuid4().hex[:8]}"
            insight = Insight(
                id=insight_id,
                type=insight_type,
                content=content,
                source_segment_ids=source_segment_ids,
                timestamp=timestamp,
                topic_id=topic_id
            )
            self.insights[insight_id] = insight
            return insight

    def get_insights_by_type(self, insight_type: InsightType) -> List[Insight]:
        """Get all insights of a given type."""
        with self.lock:
            return [i for i in self.insights.values() if i.type == insight_type]

    def get_all_insights(self) -> Dict[str, List[Insight]]:
        """Get all insights grouped by type."""
        return {
            "topics": self.get_insights_by_type(InsightType.TOPIC),
            "decisions": self.get_insights_by_type(InsightType.DECISION),
            "actions": self.get_insights_by_type(InsightType.ACTION_ITEM),
            "key_points": self.get_insights_by_type(InsightType.KEY_POINT)
        }

    def is_duplicate(self, content: str, insight_type: InsightType) -> bool:
        """Check if an insight is a duplicate based on content similarity."""
        with self.lock:
            normalized = content.lower().strip()
            for insight in self.insights.values():
                if insight.type == insight_type:
                    existing = insight.content.lower().strip()
                    if normalized == existing:
                        return True
                    words1 = set(normalized.split())
                    words2 = set(existing.split())
                    if len(words1) > 3 and len(words2) > 3:
                        overlap = len(words1 & words2) / max(len(words1), len(words2))
                        if overlap > 0.8:
                            return True
            return False


class RollingContextManager:
    """
    Maintains rolling context between analysis cycles for:
    - Smart deduplication
    - Reference resolution (understanding vague words like "that", "it")
    - Topic continuity
    """

    def __init__(self, max_context_words: int = 500):
        self.max_context_words = max_context_words
        self.context: Optional[AnalysisContext] = None
        self.lock = threading.Lock()

    def reset(self):
        """Reset context for a new session."""
        with self.lock:
            self.context = None

    def update_context(self, summary: str, topics: List[str],
                      decisions: List[str], actions: List[str],
                      last_segment_id: str, total_analyzed: int):
        """Update the rolling context after an analysis cycle."""
        with self.lock:
            self.context = AnalysisContext(
                summary=self._truncate_to_words(summary, self.max_context_words),
                recent_topics=topics[-5:],
                recent_decisions=decisions[-3:],
                recent_actions=actions[-3:],
                last_segment_id=last_segment_id,
                total_segments_analyzed=total_analyzed
            )

    def get_context(self) -> Optional[AnalysisContext]:
        """Get the current rolling context."""
        with self.lock:
            return self.context

    def get_context_prompt(self) -> str:
        """Get context formatted for LLM prompt injection."""
        with self.lock:
            if not self.context:
                return ""

            parts = []
            parts.append("=== PREVIOUS CONTEXT ===")
            parts.append(f"Summary of previous discussion:\n{self.context.summary}")

            if self.context.recent_topics:
                parts.append(f"\nRecent topics: {', '.join(self.context.recent_topics)}")

            if self.context.recent_decisions:
                parts.append(f"\nRecent decisions already recorded:\n- " +
                           "\n- ".join(self.context.recent_decisions))

            if self.context.recent_actions:
                parts.append(f"\nRecent action items already recorded:\n- " +
                           "\n- ".join(self.context.recent_actions))

            parts.append("\nIMPORTANT: Do NOT repeat any decisions or actions already listed above.")
            parts.append("=== END PREVIOUS CONTEXT ===\n")

            return "\n".join(parts)

    def _truncate_to_words(self, text: str, max_words: int) -> str:
        """Truncate text to a maximum number of words."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."


class AnalysisEngine:
    """
    The main analysis engine that coordinates all components:
    - Processes transcript in buffered segments (45-60 seconds)
    - Extracts topics, decisions, action items
    - Manages topic lifecycle
    - Maintains rolling context
    """

    def __init__(self, llm_url: str, model_id: str, provider: LLMProvider = None):
        self.brain = SingleBrainController(llm_url, model_id, provider=provider)
        self.transcript_manager = TranscriptManager()
        self.topic_engine = TopicLifecycleEngine()
        self.insight_extractor = InsightExtractor()
        self.context_manager = RollingContextManager()

        self.analysis_interval = 50
        self.min_words_for_analysis = 100

        self._analysis_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_analyzed_segment_id: Optional[str] = None
        self._words_since_last_analysis = 0

        self.lock = threading.Lock()

    def reset(self):
        """Reset all components for a new session."""
        with self.lock:
            self.transcript_manager.reset()
            self.topic_engine.reset()
            self.insight_extractor.reset()
            self.context_manager.reset()
            self._last_analyzed_segment_id = None
            self._words_since_last_analysis = 0

    def start(self):
        """Start the analysis engine."""
        # Avoid starting if already running
        if self._running:
            return
        self._running = True
        self.brain.start()
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

    def stop(self):
        """Stop the analysis engine."""
        self._running = False
        self.brain.stop()
        self._analysis_thread = None

    def add_transcript_segment(self, text: str, timestamp: str = None) -> TranscriptSegment:
        """Add a new transcript segment and trigger analysis if needed."""
        segment = self.transcript_manager.add_segment(text, timestamp)

        with self.lock:
            self._words_since_last_analysis += segment.word_count

        return segment

    def _analysis_loop(self):
        """Background loop that runs analysis when conditions are met."""
        while self._running:
            time.sleep(5)

            if not self._running:
                break

            if not self.brain.is_model_idle():
                continue

            should_analyze = False
            with self.lock:
                if self._words_since_last_analysis >= self.min_words_for_analysis:
                    should_analyze = True

            if should_analyze:
                self._run_analysis()

    def _run_analysis(self):
        """Run a full analysis cycle on unanalyzed segments."""
        segments_to_analyze = self._get_segments_for_analysis()

        if not segments_to_analyze:
            return

        if self.brain.should_interrupt_slow_track():
            return

        combined_text = " ".join([s.text for s in segments_to_analyze])
        segment_ids = [s.id for s in segments_to_analyze]
        current_time = time.time()

        context_prompt = self.context_manager.get_context_prompt()

        analysis_result = self._analyze_text(combined_text, context_prompt)

        if analysis_result:
            self._process_analysis_result(analysis_result, segment_ids, current_time)

        self.topic_engine.update_lifecycle(current_time)

        with self.lock:
            if segments_to_analyze:
                self._last_analyzed_segment_id = segments_to_analyze[-1].id
                self._words_since_last_analysis = 0

    def _get_segments_for_analysis(self) -> List[TranscriptSegment]:
        """Get segments that haven't been analyzed yet."""
        if self._last_analyzed_segment_id:
            return self.transcript_manager.get_segments_since(self._last_analyzed_segment_id)
        else:
            return self.transcript_manager.get_all_segments()

    def _analyze_text(self, text: str, context_prompt: str) -> Optional[Dict]:
        """Send text to LLM for analysis."""
        system_prompt = """You are a Meeting Analyst. Analyze the transcript segment and extract structured insights.

OUTPUT FORMAT (JSON only, no markdown):
{
    "summary": "Brief 2-3 sentence summary of this segment",
    "topics": [
        {"name": "Topic Name", "status": "active|resolved|mentioned", "resolution": "optional resolution summary"}
    ],
    "decisions": [
        {"content": "What was decided", "context": "Why/how it was decided"}
    ],
    "action_items": [
        {"content": "What needs to be done", "context": "Who mentioned it or why"}
    ],
    "key_points": [
        {"content": "Important statement or insight"}
    ],
    "reference_resolutions": {
        "vague_term": "what it refers to"
    }
}

RULES:
1. Only extract EXPLICIT decisions/actions - don't infer or assume
2. For topics: "active" = being discussed, "resolved" = concluded, "mentioned" = briefly touched
3. Resolve vague references (e.g., "that idea" → actual idea mentioned)
4. Be concise but accurate
5. Return ONLY valid JSON, no explanation or markdown"""

        user_content = f"""{context_prompt}

=== NEW SEGMENT TO ANALYZE ===
{text}
=== END SEGMENT ===

Analyze this segment and return the structured JSON output."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        result = self.brain.call_llm(messages, temperature=0.2)

        if result:
            try:
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
                    cleaned = re.sub(r'\n?```$', '', cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                print(f"Raw result: {result[:500]}")
                return None
        return None

    def _process_analysis_result(self, result: Dict, segment_ids: List[str], current_time: float):
        """Process the analysis result and update all trackers."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        for topic_data in result.get("topics", []):
            topic_name = topic_data.get("name", "").strip()
            if topic_name:
                topic = self.topic_engine.add_or_update_topic(topic_name, segment_ids[0], current_time)

                status = topic_data.get("status", "active")
                if status == "resolved":
                    self.topic_engine.resolve_topic(topic.id, topic_data.get("resolution"))

                if not self.insight_extractor.is_duplicate(topic_name, InsightType.TOPIC):
                    self.insight_extractor.add_insight(
                        InsightType.TOPIC,
                        topic_name,
                        segment_ids,
                        timestamp,
                        topic.id
                    )

        for decision in result.get("decisions", []):
            content = decision.get("content", "").strip()
            if content and not self.insight_extractor.is_duplicate(content, InsightType.DECISION):
                context_str = decision.get("context", "")
                full_content = f"{content}" + (f" ({context_str})" if context_str else "")
                self.insight_extractor.add_insight(
                    InsightType.DECISION,
                    full_content,
                    segment_ids,
                    timestamp
                )

        for action in result.get("action_items", []):
            content = action.get("content", "").strip()
            if content and not self.insight_extractor.is_duplicate(content, InsightType.ACTION_ITEM):
                context_str = action.get("context", "")
                full_content = f"{content}" + (f" ({context_str})" if context_str else "")
                self.insight_extractor.add_insight(
                    InsightType.ACTION_ITEM,
                    full_content,
                    segment_ids,
                    timestamp
                )

        for key_point in result.get("key_points", []):
            content = key_point.get("content", "").strip()
            if content and not self.insight_extractor.is_duplicate(content, InsightType.KEY_POINT):
                self.insight_extractor.add_insight(
                    InsightType.KEY_POINT,
                    content,
                    segment_ids,
                    timestamp
                )

        self._update_rolling_context(result, segment_ids)

    def _update_rolling_context(self, result: Dict, segment_ids: List[str]):
        """Update rolling context after analysis."""
        summary = result.get("summary", "")

        topics = [t.get("name", "") for t in result.get("topics", []) if t.get("name")]
        decisions = [d.get("content", "") for d in result.get("decisions", []) if d.get("content")]
        actions = [a.get("content", "") for a in result.get("action_items", []) if a.get("content")]

        existing_context = self.context_manager.get_context()
        total_analyzed = (existing_context.total_segments_analyzed if existing_context else 0) + len(segment_ids)

        self.context_manager.update_context(
            summary=summary,
            topics=topics,
            decisions=decisions,
            actions=actions,
            last_segment_id=segment_ids[-1] if segment_ids else "",
            total_analyzed=total_analyzed
        )

    def get_state(self) -> Dict:
        """Get the complete current state for UI rendering."""
        return {
            "transcript": {
                "segments": [s.to_dict() for s in self.transcript_manager.get_all_segments()],
                "full_text": self.transcript_manager.get_full_transcript()
            },
            "topics": {
                "active": [t.to_dict() for t in self.topic_engine.get_topics_by_state(TopicState.ACTIVE)],
                "resolved": [t.to_dict() for t in self.topic_engine.get_topics_by_state(TopicState.RESOLVED)],
                "parking_lot": [t.to_dict() for t in self.topic_engine.get_topics_by_state(TopicState.PARKING_LOT)]
            },
            "insights": {
                "decisions": [i.to_dict() for i in self.insight_extractor.get_insights_by_type(InsightType.DECISION)],
                "actions": [i.to_dict() for i in self.insight_extractor.get_insights_by_type(InsightType.ACTION_ITEM)],
                "key_points": [i.to_dict() for i in self.insight_extractor.get_insights_by_type(InsightType.KEY_POINT)]
            }
        }

    def get_segment_by_id(self, segment_id: str) -> Optional[TranscriptSegment]:
        """Get a specific segment for click-to-seek."""
        return self.transcript_manager.get_segment(segment_id)

    def call_llm_fast_track(self, prompt: str, context: str = "") -> str:
        """High-priority LLM call for chat (Fast Track)."""
        messages = [
            {"role": "system", "content": f"You are a meeting assistant.\n\n{context}"},
            {"role": "user", "content": prompt}
        ]
        result = self.brain.call_llm(messages, temperature=0.7)
        return result or "Error processing request."
