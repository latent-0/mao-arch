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
import json
import math
import os
import re
import time
import urllib.request


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


class EmbeddingGemmaEncoder:
    """EmbeddingGemma via a local Ollama server (on-device, local-first).

    EmbeddingGemma is Google's open 300M embedding model. Run it with
    `ollama pull embeddinggemma`; this client hits Ollama's `/api/embed` over
    stdlib urllib (no extra dependency, same pattern as the OllamaGemma
    adjudicator client). Outputs are Matryoshka-truncated to `dim` and
    L2-normalized, so 256-d node features and 768-d full vectors come from the
    same model. Used as either the trace encoder or the semantic node-feature
    encoder — fully offline, no cloud round-trip.
    """

    BATCH = 64

    def __init__(self, dim: int = 256, model: str = "embeddinggemma",
                 host: str | None = None):
        self.dim = dim
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._cache: dict[str, list[float]] = {}

    def _embed_request(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(f"{self.host}/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.load(r)
                out = []
                for v in data.get("embeddings", []):
                    v = v[:self.dim]                       # Matryoshka truncation
                    norm = math.sqrt(sum(x * x for x in v)) or 1.0
                    out.append([x / norm for x in v])
                return out
            except Exception as e:                          # transient — back off
                last_err = e
                time.sleep(2 ** attempt)
        raise last_err

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]

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
    if mode == "embeddinggemma":
        return EmbeddingGemmaEncoder(dim=dim)
    return HashingTextEncoder(dim=dim)


# Semantic node-feature encoders ------------------------------------------------
# The structural encoder can featurize a node's description with either the
# default lexical hashing (no semantic transfer to unseen step vocabulary — the
# residual gap identified in docs/experiments.md §5/§6.5) or a semantic embedder.
# Node features use a smaller default dim (256) than trace features (512).

def get_node_encoder(mode: str = "hash", dim: int | None = None):
    """Pick the node-feature encoder backend.

    mode: "hash"          — lexical hashing (default; identical to the original
                            64-d node featurizer, fully offline, no semantics)
          "embeddinggemma" — EmbeddingGemma via local Ollama (semantic, on-device)
          "gemini"         — Gemini embeddings (semantic, cloud; reachable proxy
                             for measuring the semantic-node-feature effect)
    Overridable via the MAO_NODE_ENCODER environment variable.
    """
    mode = os.environ.get("MAO_NODE_ENCODER", mode).lower()
    if mode in ("hash", "local", ""):
        return HashingTextEncoder(dim=dim or 64, seed=13)
    if mode == "embeddinggemma":
        return EmbeddingGemmaEncoder(dim=dim or 256)
    if mode == "gemini":
        return GeminiTextEncoder(dim=dim or 256)
    raise ValueError(f"unknown node-encoder mode: {mode!r}")


def encoder_mode(encoder) -> str:
    if isinstance(encoder, GeminiTextEncoder):
        return "gemini"
    if isinstance(encoder, EmbeddingGemmaEncoder):
        return "embeddinggemma"
    return "local"


def node_encoder_mode(encoder) -> str:
    """Like encoder_mode, but a plain HashingTextEncoder counts as 'hash'
    (its role as a node featurizer), not 'local'."""
    if isinstance(encoder, GeminiTextEncoder):
        return "gemini"
    if isinstance(encoder, EmbeddingGemmaEncoder):
        return "embeddinggemma"
    return "hash"
