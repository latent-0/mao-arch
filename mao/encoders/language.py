"""Language encoder: embeds an agent's reasoning/justification trace into t in R^d.

Two backends behind one interface:
  * GeminiTextEncoder  — Gemini embeddings API (used when GEMINI_API_KEY/GOOGLE_API_KEY
    is set and google-genai is installed). This is the plan's cloud path.
  * HashingTextEncoder — deterministic local n-gram hashing encoder. Zero
    dependencies, fully offline, reproducible. Used as fallback so training and
    the demo run without connectivity — consistent with the local-first story.

The encoder is *frozen*; only the MLP projector on top is trained (see joint.py).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time


class HashingTextEncoder:
    """Deterministic unigram+bigram feature hashing into R^dim, L2-normalized.

    Not a semantic model — a fixed featurizer. Sufficient signal for
    contrastive training on reasoning traces, where valid vs. violating traces
    differ in which steps/orders/artifacts they reference.
    """

    def __init__(self, dim: int = 512, seed: int = 7):
        self.dim = dim
        self.seed = seed

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(f"{self.seed}:{token}".encode()).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        return idx, sign

    def encode(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9_#]+", text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for tok in grams:
            idx, sign = self._bucket(tok)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


class GeminiTextEncoder:
    """Gemini embeddings API backend (gemini-embedding-001 family)."""

    BATCH = 64

    def __init__(self, dim: int = 512, model: str = "gemini-embedding-001"):
        from google import genai  # deferred import; optional dependency
        self._client = genai.Client()
        self.model = model
        self.dim = dim
        self._cache: dict[str, list[float]] = {}

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]

    def _embed_request(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                res = self._client.models.embed_content(
                    model=self.model, contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=self.dim))
                out = []
                for emb in res.embeddings:
                    v = list(emb.values)
                    norm = math.sqrt(sum(x * x for x in v)) or 1.0
                    out.append([x / norm for x in v])
                return out
            except Exception as e:  # rate limit / transient — back off and retry
                last_err = e
                time.sleep(2 ** attempt)
        raise last_err

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for start in range(0, len(missing), self.BATCH):
            chunk = missing[start:start + self.BATCH]
            for text, vec in zip(chunk, self._embed_request(chunk)):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]


def get_language_encoder(dim: int = 512, mode: str = "auto"):
    """Pick the language-encoder backend.

    mode: "gemini" — Gemini embeddings API (requires key; cloud path)
          "local"  — deterministic offline featurizer (local-first path)
          "auto"   — gemini iff a key is present, else local
    Overridable via the MAO_ENCODER environment variable.
    """
    mode = os.environ.get("MAO_ENCODER", mode).lower()
    if mode == "auto":
        has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        mode = "gemini" if has_key else "local"
    if mode == "gemini":
        return GeminiTextEncoder(dim=dim)
    return HashingTextEncoder(dim=dim)


def encoder_mode(encoder) -> str:
    return "gemini" if isinstance(encoder, GeminiTextEncoder) else "local"
