"""
src/llm_client.py
─────────────────
Handles communication with the local Ollama instance running Qwen 2.5 7B.

Why Ollama?
  - Easiest local LLM setup — one command to install, one to pull a model
  - Exposes a simple REST API (compatible with OpenAI's format)
  - Handles GGUF quantization automatically (4-bit works on 8GB RAM)
  - No GPU required (CPU inference is slower but works)

Model: qwen2.5:7b   (default)
  - 4-bit quantized GGUF ≈ 4.5 GB on disk, ~5–6 GB RAM at runtime
  - Excellent English + code understanding
  - Strong at following RAG-style "answer from context" instructions
"""

import json
import logging
import requests
from typing import Iterator, List, Dict, Optional

logger = logging.getLogger(__name__)

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

# OllamaClient

class OllamaClient:
    """
    Thin wrapper around the Ollama HTTP API.
    Supports both streaming and non-streaming completions.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        top_p: float = 0.9,
        num_ctx: int = 4096,    # Context window size (tokens)
        num_predict: int = 1024, # Max tokens to generate
    ):
        """
        Args:
            model:       Ollama model name (e.g. 'qwen2.5:7b', 'qwen2.5:3b').
            base_url:    Where Ollama is running.
            temperature: Lower = more factual/deterministic (0.1 is good for RAG).
            top_p:       Nucleus sampling threshold.
            num_ctx:     Context window. 4096 fits ~3 retrieved chunks + question.
            num_predict: Maximum tokens in the response.
        """
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p       = top_p
        self.num_ctx     = num_ctx
        self.num_predict = num_predict

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

    # RAG helper 

    def build_rag_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Assemble the final prompt from retrieved chunks + optional history.

        Args:
            question:         The user's current question.
            retrieved_chunks: List of {'text':..,'source_file':..,'page_number':..}
            chat_history:     Previous Q&A turns [{'role':'user','content':'..'}, ...]

        Returns:
            A fully formatted prompt string ready to send to Ollama.
        """

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source_file", "unknown")
            page   = chunk.get("page_number")
            page_str = f" (page {page})" if page else ""
            context_parts.append(
                f"[{i}] From '{source}'{page_str}:\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        history_prefix = ""
        if chat_history:
            recent = chat_history[-6:]  
            turns = []
            for msg in recent:
                role    = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:300] 
                turns.append(f"{role}: {content}")
            if turns:
                history_prefix = "RECENT CONVERSATION HISTORY:\n"
                history_prefix += "\n".join(turns)
                history_prefix += "\n\n"

        prompt = history_prefix + RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
        return prompt
