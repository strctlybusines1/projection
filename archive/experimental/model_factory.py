#!/usr/bin/env python3
"""
Model Factory — Multi-Provider LLM Abstraction
================================================

Abstracts LLM API calls behind a unified interface so the agent panel
can route each specialist agent to a different model provider.

Supported providers:
  - anthropic: Claude Sonnet 4 (best at nuanced reasoning)
  - openai:    GPT-4o mini (good at pattern matching, cheap)
  - deepseek:  DeepSeek V3 (strong at structured analysis, cheap)

Following Moon Dev's ModelFactory pattern from Polymarket agent.

Usage:
    from model_factory import ModelFactory

    model = ModelFactory.create('anthropic')
    response = model.generate(system_prompt, user_content)

    # Or use the convenience function:
    response = call_model('openai', system_prompt, user_content)

API Keys (from .env or environment):
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    DEEPSEEK_API_KEY=sk-...
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict
from pathlib import Path

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass


class BaseModel(ABC):
    """Abstract base for all LLM providers."""

    provider: str = 'base'
    model_name: str = ''
    cost_per_1k_input: float = 0.0   # $ per 1K input tokens
    cost_per_1k_output: float = 0.0  # $ per 1K output tokens

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout: int = 60,
    ) -> Optional[str]:
        """Generate a response. Returns raw text or None on failure."""
        pass

    def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout: int = 60,
    ) -> Optional[Dict]:
        """Generate and parse JSON response. Handles markdown fences."""
        text = self.generate(system_prompt, user_content, temperature, max_tokens, timeout)
        if not text:
            return None

        # Strip markdown fences
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
            text = text.rsplit('```', 1)[0]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            print(f"    ⚠ {self.provider}: Failed to parse JSON response")
            return None


class AnthropicModel(BaseModel):
    """Claude via Anthropic API."""

    provider = 'anthropic'
    model_name = 'claude-sonnet-4-20250514'
    cost_per_1k_input = 0.003
    cost_per_1k_output = 0.015

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        if model:
            self.model_name = model

    def generate(self, system_prompt, user_content, temperature=0.3,
                 max_tokens=4000, timeout=60) -> Optional[str]:
        if not self.api_key:
            print("    ⚠ Anthropic: No API key")
            return None
        try:
            import requests
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key,
                    'anthropic-version': '2023-06-01',
                },
                json={
                    'model': self.model_name,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_content}],
                },
                timeout=timeout,
            )
            data = response.json()
            if 'content' in data and data['content']:
                return data['content'][0]['text']
            else:
                error = data.get('error', {}).get('message', str(data))
                print(f"    ⚠ Anthropic API error: {error}")
                return None
        except Exception as e:
            print(f"    ⚠ Anthropic call failed: {e}")
            return None


class OpenAIModel(BaseModel):
    """GPT via OpenAI API."""

    provider = 'openai'
    model_name = 'gpt-4o-mini'
    cost_per_1k_input = 0.00015
    cost_per_1k_output = 0.0006

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        if model:
            self.model_name = model

    def generate(self, system_prompt, user_content, temperature=0.3,
                 max_tokens=4000, timeout=60) -> Optional[str]:
        if not self.api_key:
            print("    ⚠ OpenAI: No API key")
            return None
        try:
            import requests
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                },
                json={
                    'model': self.model_name,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_content},
                    ],
                },
                timeout=timeout,
            )
            data = response.json()
            if 'choices' in data and data['choices']:
                return data['choices'][0]['message']['content']
            else:
                error = data.get('error', {}).get('message', str(data))
                print(f"    ⚠ OpenAI API error: {error}")
                return None
        except Exception as e:
            print(f"    ⚠ OpenAI call failed: {e}")
            return None


class DeepSeekModel(BaseModel):
    """DeepSeek V3 via DeepSeek API (OpenAI-compatible)."""

    provider = 'deepseek'
    model_name = 'deepseek-chat'
    cost_per_1k_input = 0.00055
    cost_per_1k_output = 0.00166

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', '')
        if model:
            self.model_name = model

    def generate(self, system_prompt, user_content, temperature=0.3,
                 max_tokens=4000, timeout=60) -> Optional[str]:
        if not self.api_key:
            print("    ⚠ DeepSeek: No API key")
            return None
        try:
            import requests
            response = requests.post(
                'https://api.deepseek.com/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                },
                json={
                    'model': self.model_name,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_content},
                    ],
                },
                timeout=timeout,
            )
            data = response.json()
            if 'choices' in data and data['choices']:
                return data['choices'][0]['message']['content']
            else:
                error = data.get('error', {}).get('message', str(data))
                print(f"    ⚠ DeepSeek API error: {error}")
                return None
        except Exception as e:
            print(f"    ⚠ DeepSeek call failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

class ModelFactory:
    """Create model instances by provider name."""

    PROVIDERS = {
        'anthropic': AnthropicModel,
        'claude': AnthropicModel,
        'openai': OpenAIModel,
        'gpt': OpenAIModel,
        'deepseek': DeepSeekModel,
    }

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseModel:
        """Create a model instance.

        Args:
            provider: 'anthropic', 'openai', or 'deepseek'
            **kwargs: Passed to model constructor (api_key, model)
        """
        provider = provider.lower()
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. "
                             f"Available: {list(cls.PROVIDERS.keys())}")
        return cls.PROVIDERS[provider](**kwargs)

    @classmethod
    def available(cls) -> Dict[str, bool]:
        """Check which providers have API keys configured."""
        status = {}
        for name, model_cls in [('anthropic', AnthropicModel),
                                 ('openai', OpenAIModel),
                                 ('deepseek', DeepSeekModel)]:
            instance = model_cls()
            status[name] = bool(instance.api_key)
        return status


def call_model(provider: str, system_prompt: str, user_content: str,
               **kwargs) -> Optional[str]:
    """Convenience function: create model and call in one step."""
    model = ModelFactory.create(provider)
    return model.generate(system_prompt, user_content, **kwargs)


def call_model_json(provider: str, system_prompt: str, user_content: str,
                    **kwargs) -> Optional[Dict]:
    """Convenience function: create model, call, parse JSON."""
    model = ModelFactory.create(provider)
    return model.generate_json(system_prompt, user_content, **kwargs)


# ═══════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Model Factory — Provider Status")
    print("=" * 40)
    status = ModelFactory.available()
    for provider, has_key in status.items():
        icon = "✅" if has_key else "❌"
        print(f"  {icon} {provider}")

    # Quick test with available providers
    print("\nQuick test (asking each model to say hello):")
    for provider, has_key in status.items():
        if has_key:
            model = ModelFactory.create(provider)
            t0 = time.time()
            resp = model.generate(
                "You are a test assistant. Respond in exactly 5 words.",
                "Say hello and identify yourself.",
                max_tokens=50,
            )
            elapsed = time.time() - t0
            print(f"  {provider} ({elapsed:.1f}s): {resp}")
        else:
            print(f"  {provider}: skipped (no key)")
