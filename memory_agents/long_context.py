import os
import time
from typing import Optional

import tiktoken
from openai import OpenAI

from utils.templates import SYSTEM_MESSAGE


def _create_oai_client():
    """OpenAI client with optional Azure endpoint."""
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=azure_endpoint,
        )
    return OpenAI()


class BaseLongContextMemory:
    """Minimal long-context memory interface."""

    def __init__(
        self,
        model: str,
        max_context_tokens: int = 120_000,
        max_output_tokens: int = 256,
        tokenizer_model: Optional[str] = None,
    ):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.system_message = SYSTEM_MESSAGE
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model or "gpt-4o-mini")
        self.context = ""

    def add_chunk(self, chunk: str):
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {chunk}"
        self.context = (self.context + "\n" + stamped).strip()
        self._truncate_context()

    def _truncate_context(self):
        tokens = self.tokenizer.encode(self.context, disallowed_special=())
        if len(tokens) > self.max_context_tokens:
            self.context = self.tokenizer.decode(tokens[-self.max_context_tokens :])

    def _build_prompt(self, question: str) -> str:
        return (
            "Use only the memorized context below to answer concisely.\n\n"
            f"Context:\n{self.context}\n\nQuestion: {question}\nAnswer:"
        )

    def answer(self, question: str) -> str:
        prompt = self._build_prompt(question)
        return self._run_chat(prompt)

    def wrap_user_prompt(self, question: str) -> str:
        """Return a user prompt augmented with retrieved context (simple concat)."""
        return f"Context:\n{self.context}\n\nUser Prompt:\n{question}"

    def act(self, prompt: str) -> str:
        """Agent behavior: respond to an already-wrapped prompt."""
        return self._run_chat(prompt)

    # To be implemented by subclasses
    def _run_chat(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIChatMemory(BaseLongContextMemory):
    """Long-context memory using OpenAI/Azure Chat Completions."""

    def __init__(self, model: str, max_context_tokens: int = 120_000, max_output_tokens: int = 256):
        super().__init__(
            model=model,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            tokenizer_model=model if ("gpt" in model or "o4" in model) else "gpt-4o-mini",
        )
        self.client = _create_oai_client()

    def _run_chat(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_output_tokens,
        )
        return resp.choices[0].message.content


class AnthropicLongContextMemory(BaseLongContextMemory):
    """Long-context memory using Anthropic Claude."""

    def __init__(self, model: str = "claude-3-7-sonnet-20250219", max_context_tokens: int = 120_000, max_output_tokens: int = 256):
        super().__init__(model=model, max_context_tokens=max_context_tokens, max_output_tokens=max_output_tokens)
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("Anthropic_API_KEY"))

    def _run_chat(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_output_tokens,
        )
        return resp.content[0].text


class GeminiLongContextMemory(BaseLongContextMemory):
    """Long-context memory using Google Gemini."""

    def __init__(self, model: str = "gemini-2.0-flash", max_context_tokens: int = 120_000, max_output_tokens: int = 256):
        super().__init__(model=model, max_context_tokens=max_context_tokens, max_output_tokens=max_output_tokens)
        from google import genai

        self.client = genai.Client(api_key=os.environ.get("Google_API_KEY"))
        self.max_output_tokens = max_output_tokens

    def _run_chat(self, prompt: str) -> str:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_message,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        return resp.text


class LongContextGPT4oMini(OpenAIChatMemory):
    def __init__(self):
        super().__init__("gpt-4o-mini")


class LongContextGPT4o(OpenAIChatMemory):
    def __init__(self):
        super().__init__("gpt-4o")


class LongContextGPT41Mini(OpenAIChatMemory):
    def __init__(self):
        super().__init__("gpt-4.1-mini")


class LongContextO4Mini(OpenAIChatMemory):
    def __init__(self):
        super().__init__("o4-mini")


class LongContextClaudeSonnet(AnthropicLongContextMemory):
    def __init__(self):
        super().__init__("claude-3-7-sonnet-20250219")


class LongContextGeminiFlash(GeminiLongContextMemory):
    def __init__(self):
        super().__init__("gemini-2.0-flash")
