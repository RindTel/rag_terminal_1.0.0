"""
src/llm_client.py
─────────────────
LLM clients for the RAG pipeline, plus the shared prompt/token-budget logic.

Two interchangeable backends implement the same surface, selected by
config.LLM_BACKEND (see create_llm_client):

  - OllamaClient   — talks to a local Ollama daemon over HTTP. The dev default,
                     and what the eval harness + tests exercise.
  - LlamaCppClient — loads a GGUF in-process via llama-cpp-python. The packaged
                     desktop app's engine: no external daemon, no HTTP, nothing
                     to install. Auto-selected when the app is frozen.

Both share _RagPromptMixin (assemble/prompt_budget/rewrite-cleaning), because the
prompt construction and token budgeting are backend-agnostic — they depend only
on num_ctx/num_predict and the retrieved chunks, not on how tokens are generated.

Model: qwen2.5 (7b via Ollama in dev, 3b GGUF in the bundle)
  - Excellent English + code understanding
  - Strong at following RAG-style "answer from context" instructions
"""

import os
import re
import json
import math
import logging
import requests
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Optional

import config

logger = logging.getLogger(__name__)

# Token budgeting
#
# Ollama truncates an over-long prompt FROM THE LEFT — silently. That eats the
# chat history and then the HIGHEST-ranked context chunks, which are exactly the
# ones we most want the model to see. So we must know the prompt's token cost
# BEFORE sending it, and drop the lowest-ranked chunks ourselves.
#
# There is no way to count Qwen tokens exactly without a round-trip:
#   - Ollama exposes no /api/tokenize (404).
#   - `num_predict: 0` is ignored — it generates anyway, so there is no dry-run.
#   - `num_predict: 1` does return an exact prompt_eval_count, but that is one
#     HTTP call per count; a packing loop would need one per chunk.
#   - The model is user-switchable (qwen/llama/mistral), so pinning one
#     tokenizer would be wrong for the others.
#
# So we estimate locally, biased to OVER-count. An under-count would recreate
# the exact silent truncation we are fixing, so the bias must always be upward.
# Calibrated against real prompt_eval_count on this corpus:
#     prose            4.17 - 4.75 chars/token
#     ASCII art / code 3.72 chars/token   <- worst case
# 3.5 sits below the worst observed ratio, so it over-counts on every sample.

CHARS_PER_TOKEN = 3.5

# Ollama wraps the prompt in the model's chat template (<|im_start|>system ...).
# Those tokens count against num_ctx but never appear in our text: an 18-char
# prompt reports 34 tokens. Measured ~28; rounded up.
CHAT_TEMPLATE_OVERHEAD_TOKENS = 48

# Final cushion against estimator drift on unusual input (dense CJK, base64...).
SAFETY_MARGIN_TOKENS = 96


def estimate_tokens(text: str) -> int:
    """
    Conservative token count. Deliberately over-estimates — see above.
    Never let this under-count; that is the bug, not a rounding detail.
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass
class PromptBuild:
    """A prompt plus an honest account of what had to be left out to fit."""
    prompt: str
    used_chunks: List[Dict] = field(default_factory=list)
    dropped_chunks: List[Dict] = field(default_factory=list)
    est_tokens: int = 0
    budget: int = 0
    history_dropped: bool = False

    @property
    def truncated(self) -> bool:
        return bool(self.dropped_chunks) or self.history_dropped


# Prompt templates

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided document excerpts.

RULES:
1. Answer ONLY using the information in the CONTEXT section below.
2. If the context does not contain enough information to answer, say:
   "I couldn't find a clear answer in the uploaded documents. Please try rephrasing or upload more relevant documents."
3. Be concise and accurate.
4. When citing specific facts, mention which document they come from.
5. Never make up information that isn't in the context."""

RAG_PROMPT_TEMPLATE = """CONTEXT (excerpts from uploaded documents):
──────────────────────────────────────────
{context}
──────────────────────────────────────────

QUESTION: {question}

Answer based only on the context above:"""


# Shared prompt/budget logic (backend-agnostic)

class _RagPromptMixin:
    """
    Prompt assembly + token budgeting shared by every backend.

    Depends only on self.num_ctx / self.num_predict and the retrieved chunks, so
    it is identical whether tokens come from Ollama over HTTP or from an
    in-process GGUF. Both OllamaClient and LlamaCppClient inherit it.
    """

    @staticmethod
    def _clean_rewrite(text: str) -> str:
        """First non-empty line, stripped of quotes and a leading 'Standalone…:' echo."""
        line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        line = re.sub(r'^(standalone( question)?|rewritten( question)?)\s*[:\-]\s*',
                      '', line, flags=re.I).strip()
        return line.strip('"\'' + '`').strip()

    # RAG helper

    def prompt_budget(self) -> int:
        """
        Tokens actually available for the prompt.

        num_ctx is the WHOLE window — prompt AND generation share it. Reserving
        num_predict is not optional: without it, a prompt that fits num_ctx
        leaves no room to answer.
        """
        return (
            self.num_ctx
            - self.num_predict
            - CHAT_TEMPLATE_OVERHEAD_TOKENS
            - SAFETY_MARGIN_TOKENS
        )

    @staticmethod
    def _format_chunk(i: int, chunk: Dict) -> str:
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_number")
        page_str = f" (page {page})" if page else ""
        return f"[{i}] From '{source}'{page_str}:\n{chunk['text']}"

    @staticmethod
    def _format_history(chat_history: Optional[List[Dict]]) -> str:
        if not chat_history:
            return ""
        turns = []
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            turns.append(f"{role}: {msg['content'][:300]}")
        if not turns:
            return ""
        return "RECENT CONVERSATION HISTORY:\n" + "\n".join(turns) + "\n\n"

    def assemble(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        chat_history: Optional[List[Dict]] = None,
    ) -> PromptBuild:
        """
        Build the prompt within the token budget.

        Chunks arrive in RANK ORDER (highest similarity first — VectorStore.search
        sorts descending). We pack from the top and stop when the budget is spent,
        so the chunks that get dropped are always the LOWEST-ranked ones. Left to
        Ollama, the opposite happens: it truncates from the left and eats the best
        chunks first.

        Priority when space runs out:  context chunks (by rank) > chat history.
        """
        budget = self.prompt_budget()

        history = self._format_history(chat_history)
        scaffold = estimate_tokens(RAG_PROMPT_TEMPLATE.format(context="", question=question))
        fixed = estimate_tokens(SYSTEM_PROMPT) + scaffold

        def build(used: List[Dict], hist: str) -> str:
            context = "\n\n".join(self._format_chunk(i, c) for i, c in enumerate(used, 1))
            return hist + RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # History is the first thing sacrificed — it is conversational polish,
        # whereas the context chunks are the entire basis for a grounded answer.
        history_dropped = False
        if retrieved_chunks and fixed + estimate_tokens(history) >= budget:
            logger.warning(
                "Prompt budget (%d) exhausted before any context chunk fit — dropping chat history.",
                budget,
            )
            history, history_dropped = "", True

        spent = fixed + estimate_tokens(history)
        used: List[Dict] = []
        dropped: List[Dict] = []

        for chunk in retrieved_chunks:
            cost = estimate_tokens(self._format_chunk(len(used) + 1, chunk)) + 2  # +2 for the "\n\n" joiner
            if spent + cost <= budget:
                used.append(chunk)
                spent += cost
            else:
                dropped.append(chunk)

        # A single chunk larger than the whole budget would otherwise yield an
        # EMPTY context — worse than a partial one, because the model then
        # answers ungrounded. Truncate it instead.
        if retrieved_chunks and not used:
            room_chars = max(0, int((budget - spent) * CHARS_PER_TOKEN))
            head = dict(retrieved_chunks[0])
            head["text"] = head["text"][:room_chars].rstrip()
            if head["text"]:
                logger.warning(
                    "Top-ranked chunk alone exceeds the prompt budget — truncating it to fit."
                )
                used, dropped = [head], list(retrieved_chunks[1:])
                spent += estimate_tokens(self._format_chunk(1, head))

        if dropped:
            logger.warning(
                "Prompt budget: kept %d/%d chunk(s), dropped %d lowest-ranked "
                "(~%d/%d est. tokens). Raise num_ctx or lower top_k to fit more.",
                len(used), len(retrieved_chunks), len(dropped), spent, budget,
            )

        return PromptBuild(
            prompt=build(used, history),
            used_chunks=used,
            dropped_chunks=dropped,
            est_tokens=spent,
            budget=budget,
            history_dropped=history_dropped,
        )

    def build_rag_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """Budget-aware prompt string. Thin wrapper over assemble()."""
        return self.assemble(question, retrieved_chunks, chat_history).prompt


# OllamaClient

class OllamaClient(_RagPromptMixin):
    """
    Thin wrapper around the Ollama HTTP API.
    Supports both streaming and non-streaming completions.
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = None,
        top_p: float = None,
        num_ctx: int = None,      # Context window size (tokens)
        num_predict: int = None,  # Max tokens to generate
    ):
        """
        Args:
            model:       Ollama model name (e.g. 'qwen2.5:7b', 'qwen2.5:3b').
            base_url:    Where Ollama is running.
            temperature: Lower = more factual/deterministic (0.1 is good for RAG).
            top_p:       Nucleus sampling threshold.
            num_ctx:     Total context window — prompt AND generation share it.
                         prompt_budget() derives the real prompt limit from
                         num_ctx minus num_predict.
            num_predict: Maximum tokens in the response.

        Every argument defaults to config.py (which honours env vars).
        """
        self.model       = model if model is not None else config.OLLAMA_MODEL
        self.base_url    = (base_url if base_url is not None else config.OLLAMA_BASE_URL).rstrip("/")
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.top_p       = top_p if top_p is not None else config.LLM_TOP_P
        self.num_ctx     = num_ctx if num_ctx is not None else config.LLM_NUM_CTX
        self.num_predict = num_predict if num_predict is not None else config.LLM_NUM_PREDICT

    # Health check

    def is_available(self) -> bool:
        """Ping Ollama to check if it's running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def is_model_pulled(self) -> bool:
        """Check if the selected model is already downloaded."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(m["name"].startswith(self.model.split(":")[0]) for m in models)
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return names of all models available in this Ollama instance."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    # Core generation

    def generate(
        self,
        prompt: str,
        stream: bool = False,
    ) -> str:
        """
        Non-streaming generation using /api/generate.
        Returns the full response text.
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p":       self.top_p,
                "num_ctx":     self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,   # Allow up to 2 minutes for slow CPU inference
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure it's running:\n"
                "  ollama serve"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                "Ollama took too long to respond. "
                "Try a smaller model like 'qwen2.5:3b' or reduce num_predict."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Streaming generation — yields text tokens as they arrive.
        Use this in Streamlit with st.write_stream() for a typing effect.
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p":       self.top_p,
                "num_ctx":     self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=180,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError:
            yield "\n\n❌ Cannot connect to Ollama. Run `ollama serve` in a terminal."
        except requests.exceptions.Timeout:
            yield "\n\n❌ Generation timed out. Try a smaller model or shorter context."

    # Query rewriting

    def rewrite_query(self, question: str, chat_history: List[Dict]) -> str:
        """
        Rewrite a conversational follow-up into a standalone query using the last
        couple of turns. Returns the rewritten query, or the ORIGINAL question if
        anything looks off — a bad rewrite must never hurt retrieval more than the
        raw query would.

        Deterministic (temperature 0, fixed seed) so eval runs stay reproducible.
        """
        recent = (chat_history or [])[-4:]   # last ~2 turns
        convo = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent
        )
        prompt = (
            "Rewrite the user's follow-up question into a standalone question that "
            "makes sense on its own, using the conversation for context. Resolve "
            "pronouns and references (it, they, that, ...) to what they refer to. "
            "Output ONLY the rewritten question on a single line — no preamble, no "
            "quotes. If the question is already standalone, output it unchanged.\n\n"
            f"Conversation:\n{convo}\n\n"
            f"Follow-up: {question}\n"
            "Standalone question:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 0, "num_predict": 64,
                        "num_ctx": self.num_ctx},
        }
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
        except Exception as e:
            logger.warning("Query rewrite failed (%s) — using the original query.", e)
            return question

        rewritten = self._clean_rewrite(text)

        # Guard rails: empty or implausibly long → distrust it, keep the original.
        if not rewritten or len(rewritten) > max(80, len(question) * 4):
            logger.info("Query rewrite discarded (implausible) — using original.")
            return question

        if rewritten.lower() != question.lower():
            logger.info("Rewrote query %r -> %r", question, rewritten)
        return rewritten


# LlamaCppClient

class LlamaCppClient(_RagPromptMixin):
    """
    In-process LLM backend using llama-cpp-python — the packaged app's engine.

    Loads a GGUF model directly: no Ollama daemon, no HTTP, nothing to install.
    The model is memory-mapped LAZILY on first generation so construction and
    health checks stay cheap — a 2 GB mmap must never happen just to render the
    sidebar status.

    Exposes the same surface RAGPipeline/UI rely on (is_available, is_model_pulled,
    list_models, generate, generate_stream, rewrite_query) plus the shared
    prompt/budget logic from _RagPromptMixin.
    """

    def __init__(
        self,
        model: str = None,
        model_path: str = None,
        temperature: float = None,
        top_p: float = None,
        num_ctx: int = None,
        num_predict: int = None,
    ):
        self.model_path  = model_path if model_path is not None else config.LLAMACPP_MODEL_PATH
        # Display name for the UI/status — derived from the file, not a daemon tag.
        self.model       = model if model is not None else os.path.basename(self.model_path)
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.top_p       = top_p if top_p is not None else config.LLM_TOP_P
        self.num_ctx     = num_ctx if num_ctx is not None else config.LLM_NUM_CTX
        self.num_predict = num_predict if num_predict is not None else config.LLM_NUM_PREDICT
        self._llama = None

    # Model loading (lazy)

    def _get_llama(self):
        if self._llama is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is required for the llamacpp backend.\n"
                    "Install with:  pip install llama-cpp-python"
                )
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"GGUF model not found at '{self.model_path}'. Set "
                    "LLAMACPP_MODEL_PATH or bundle the model under resources/models/."
                )
            logger.info("Loading GGUF model '%s' (n_ctx=%d) ...", self.model_path, self.num_ctx)
            self._llama = Llama(
                model_path=self.model_path,
                n_ctx=self.num_ctx,
                verbose=False,
            )
            logger.info("llama.cpp model ready.")
        return self._llama

    # Health checks

    def is_available(self) -> bool:
        """Available iff the GGUF file is on disk. Does NOT load the model."""
        return os.path.exists(self.model_path)

    def is_model_pulled(self) -> bool:
        """Same signal as is_available — the model IS the bundled file."""
        return os.path.exists(self.model_path)

    def list_models(self) -> List[str]:
        return [self.model]

    # Core generation

    def _messages(self, prompt: str) -> List[Dict]:
        # create_chat_completion applies the GGUF's own chat template, matching
        # how Ollama wraps `system` + `prompt`.
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

    def generate(self, prompt: str, stream: bool = False) -> str:
        """Non-streaming generation. Returns the full response text."""
        llama = self._get_llama()
        out = llama.create_chat_completion(
            messages=self._messages(prompt),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.num_predict,
            stream=False,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Streaming generation — yields text tokens as they arrive."""
        try:
            llama = self._get_llama()
        except Exception as e:
            # Mirror OllamaClient.generate_stream: surface the failure as text
            # rather than raising inside the Streamlit stream.
            yield f"\n\n❌ {e}"
            return
        for chunk in llama.create_chat_completion(
            messages=self._messages(prompt),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.num_predict,
            stream=True,
        ):
            token = chunk["choices"][0].get("delta", {}).get("content", "")
            if token:
                yield token

    # Query rewriting (identical guard rails to the Ollama path)

    def rewrite_query(self, question: str, chat_history: List[Dict]) -> str:
        """
        Rewrite a conversational follow-up into a standalone query. Returns the
        ORIGINAL question if anything looks off — a bad rewrite must never hurt
        retrieval more than the raw query would. Deterministic (temp 0, seed 0).
        """
        recent = (chat_history or [])[-4:]
        convo = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent
        )
        prompt = (
            "Rewrite the user's follow-up question into a standalone question that "
            "makes sense on its own, using the conversation for context. Resolve "
            "pronouns and references (it, they, that, ...) to what they refer to. "
            "Output ONLY the rewritten question on a single line — no preamble, no "
            "quotes. If the question is already standalone, output it unchanged.\n\n"
            f"Conversation:\n{convo}\n\n"
            f"Follow-up: {question}\n"
            "Standalone question:"
        )
        try:
            llama = self._get_llama()
            out = llama.create_completion(
                prompt=prompt, temperature=0.0, seed=0, max_tokens=64,
            )
            text = (out["choices"][0]["text"] or "").strip()
        except Exception as e:
            logger.warning("Query rewrite failed (%s) — using the original query.", e)
            return question

        rewritten = self._clean_rewrite(text)
        if not rewritten or len(rewritten) > max(80, len(question) * 4):
            logger.info("Query rewrite discarded (implausible) — using original.")
            return question
        if rewritten.lower() != question.lower():
            logger.info("Rewrote query %r -> %r", question, rewritten)
        return rewritten


# Backend factory

def create_llm_client(model: str = None, backend: str = None, **kwargs):
    """
    Build the LLM client for the configured backend.

    backend defaults to config.LLM_BACKEND ('ollama' in dev, 'llamacpp' when
    frozen). Signature-compatible with the old OllamaClient(model=...) call, so
    RAGPipeline needs only a one-line change.
    """
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == "llamacpp":
        return LlamaCppClient(model=model, **kwargs)
    if backend == "ollama":
        return OllamaClient(model=model, **kwargs)
    raise ValueError(
        f"Unknown LLM_BACKEND {backend!r} (expected 'ollama' or 'llamacpp')."
    )
