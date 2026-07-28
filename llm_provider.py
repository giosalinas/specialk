"""
LLM Provider Abstraction Layer

Supports multiple LLM backends:
- Local: OpenAI-compatible server (LM Studio, Ollama, etc.)
- Claude: Anthropic Claude API

Usage:
    provider = get_provider(settings)
    result = provider.chat(system_prompt, user_message, temperature=0.3)
"""

import os
import re
import json
import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
        """Send a chat request and return the response text."""
        pass

    @abstractmethod
    def chat_with_messages(self, messages: List[Dict],
                           temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
        """Send a chat request with a full messages list (OpenAI format)."""
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for display."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for display."""
        pass


class LocalLLMProvider(LLMProvider):
    """OpenAI-compatible local LLM (LM Studio, Ollama, etc.)."""

    def __init__(self, base_url: str = "http://localhost:1234/v1/chat/completions",
                 model_id: str = "local-model"):
        self.base_url = base_url
        self.model_id = model_id

    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.chat_with_messages(messages, temperature, timeout)

    def chat_with_messages(self, messages: List[Dict],
                           temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
        try:
            payload = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "stream": False
            }
            response = requests.post(self.base_url, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                print(f"Local LLM Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Local LLM Connection Error: {e}")
            return None

    def provider_name(self) -> str:
        return "Local LLM"

    def model_name(self) -> str:
        return self.model_id


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider."""

    MODELS = {
        "claude-sonnet-4-20250514": "Claude Sonnet 4",
        "claude-haiku-4-20250414": "Claude Haiku 4",
        "claude-opus-4-20250414": "Claude Opus 4",
    }

    def __init__(self, api_key: str = None, model_id: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model_id = model_id
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.api_version = "2023-06-01"

        if not self.api_key:
            print("WARNING: No Anthropic API key set. Set ANTHROPIC_API_KEY env var or pass api_key.")

    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "content-type": "application/json"
            }

            payload = {
                "model": self.model_id,
                "max_tokens": 4096,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}]
            }

            response = requests.post(self.base_url, json=payload,
                                     headers=headers, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text'].strip()
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get('error', {}).get('message', response.text)
                print(f"Claude API Error {response.status_code}: {error_msg}")
                return None
        except Exception as e:
            print(f"Claude API Connection Error: {e}")
            return None

    def chat_with_messages(self, messages: List[Dict],
                           temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
        """Convert OpenAI-format messages to Claude format and send."""
        system_prompt = ""
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Claude requires at least one user message
        if not claude_messages:
            claude_messages = [{"role": "user", "content": "Hello"}]

        # Claude doesn't allow consecutive messages from the same role
        merged = []
        for msg in claude_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)

        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "content-type": "application/json"
            }

            payload = {
                "model": self.model_id,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": merged
            }

            if system_prompt.strip():
                payload["system"] = system_prompt.strip()

            response = requests.post(self.base_url, json=payload,
                                     headers=headers, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text'].strip()
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get('error', {}).get('message', response.text)
                print(f"Claude API Error {response.status_code}: {error_msg}")
                return None
        except Exception as e:
            print(f"Claude API Connection Error: {e}")
            return None

    def provider_name(self) -> str:
        return "Claude"

    def model_name(self) -> str:
        return self.MODELS.get(self.model_id, self.model_id)


# --- Factory ---

def get_provider(settings: dict, purpose: str = "chat") -> LLMProvider:
    """
    Create an LLM provider based on settings.

    Args:
        settings: App settings dict from load_settings()
        purpose: "chat" for main chat/analysis, "transcription" for transcript formatting
    """
    if purpose == "transcription":
        provider_type = settings.get("llm_transcription_provider", "local")
        model_id = settings.get("llm_transcription_model_id", "local-model")
    else:
        provider_type = settings.get("llm_provider", "local")
        model_id = settings.get("llm_model_id", "local-model")

    if provider_type == "claude":
        api_key = settings.get("claude_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        claude_model = settings.get("claude_model_id", "claude-sonnet-4-20250514")

        if purpose == "transcription":
            claude_model = settings.get("claude_transcription_model_id", claude_model)

        return ClaudeProvider(api_key=api_key, model_id=claude_model)
    else:
        local_url = settings.get("llm_local_url", "http://localhost:1234/v1/chat/completions")
        return LocalLLMProvider(base_url=local_url, model_id=model_id)


def test_provider(provider: LLMProvider) -> tuple[bool, str]:
    """Test if a provider is working. Returns (success, message)."""
    try:
        start = time.time()
        result = provider.chat(
            system_prompt="You are a test assistant.",
            user_message="Reply with exactly: OK",
            temperature=0.0,
            timeout=15
        )
        elapsed = round(time.time() - start, 2)

        if result:
            return True, f"✅ {provider.provider_name()} ({provider.model_name()}) responded in {elapsed}s"
        else:
            return False, f"❌ {provider.provider_name()} returned empty response"
    except Exception as e:
        return False, f"❌ {provider.provider_name()} error: {e}"
