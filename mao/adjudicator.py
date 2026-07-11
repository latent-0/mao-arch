"""Local adjudicator — the on-device handoff safety gate (Gemma track).

The gate itself is a cosine-similarity check over fixed-size vectors in the
joint space: constant cost regardless of how long the orchestration has run,
zero cloud round-trips. Three explicit outcomes:

    APPROVE          alignment >= tau_hi           -> action proceeds
    FLAG_TO_HUMAN    tau_lo <= alignment < tau_hi  -> surfaced for review
    REQUEST_REPLAN   alignment < tau_lo            -> routed back with the
                                                      specific violated
                                                      constraint attached

A local Gemma model (via Ollama) optionally writes the natural-language replan
message; when no local model is available a template is used. The *decision*
never depends on any model call — it is a threshold on a learned metric,
quantitative and inspectable.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from enum import Enum

import torch

from .encoders.language import get_language_encoder, get_node_encoder
from .graph import TaskGraph, Violation
from .handoff import HandoffPacket
from .joint import JointModel

ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "artifacts")


def artifact_subdir(mode: str, node_mode: str = "hash", holdout=None) -> str:
    """Artifact directory name for a (trace-encoder mode, node-encoder mode,
    holdout) combination. Semantic node features get a `_snode-<mode>` suffix so
    they never clobber the lexical-node artifacts. Kept here so train, eval, and
    the adjudicator all agree on the layout."""
    base = mode if holdout is None else f"{mode}_holdout{holdout}"
    return base if node_mode in (None, "hash") else f"{base}_snode-{node_mode}"


class Verdict(str, Enum):
    APPROVE = "approve"
    FLAG_TO_HUMAN = "flag_to_human"
    REQUEST_REPLAN = "request_replan"


@dataclass
class Decision:
    verdict: Verdict
    alignment: float              # cos(f(g), f(t)) for the proposed action
    tau_lo: float
    tau_hi: float
    nearest_violation_sim: float  # max cosine to the known-violating bank
    violations: list[Violation] = field(default_factory=list)
    message: str = ""

    def as_dict(self) -> dict:
        return {"verdict": self.verdict.value, "alignment": round(self.alignment, 4),
                "band": [round(self.tau_lo, 4), round(self.tau_hi, 4)],
                "nearest_violation_sim": round(self.nearest_violation_sim, 4),
                "violations": [v.detail for v in self.violations],
                "message": self.message}


class OllamaGemma:
    """Minimal Ollama HTTP client for local Gemma explanation generation."""

    def __init__(self, model: str | None = None, host: str = "http://localhost:11434"):
        self.host = host
        self.model = model or self._autodetect()

    def _autodetect(self) -> str | None:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                tags = json.load(r)
            names = [m["name"] for m in tags.get("models", [])]
            gemmas = [n for n in names if "gemma" in n.lower()]
            return gemmas[0] if gemmas else None
        except Exception:
            return None

    def generate(self, prompt: str) -> str | None:
        if not self.model:
            return None
        try:
            # Gemma 4 is a thinking model: thinking is disabled here for
            # latency (the gate decision never depends on this call), and
            # keep_alive keeps the model warm between adjudications.
            body = json.dumps({"model": self.model, "stream": False, "think": False,
                               "messages": [{"role": "user", "content": prompt}],
                               "options": {"temperature": 0.2, "num_predict": 180},
                               "keep_alive": "30m"}).encode()
            req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                msg = json.load(r).get("message", {})
                return (msg.get("content") or "").strip() or None
        except Exception:
            return None


class Adjudicator:
    def __init__(self, model: JointModel, language_encoder,
                 tau_lo: float, tau_hi: float,
                 negative_bank: torch.Tensor | None = None,
                 use_local_llm: bool = True):
        self.model = model
        self.encoder = language_encoder
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.negative_bank = negative_bank
        self.gemma = OllamaGemma() if use_local_llm else None

    @classmethod
    def from_artifacts(cls, mode: str = "auto", artifact_root: str = ARTIFACT_ROOT,
                       use_local_llm: bool = True) -> "Adjudicator":
        """Load the adjudicator for an encoder mode ("local"|"gemini"|"auto").

        The encoder is reconstructed to match the one the artifacts were
        trained with — mixing encoders across the joint space is invalid.
        """
        if mode == "auto":
            has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            mode = "gemini" if has_key else "local"
            if not os.path.isdir(os.path.join(artifact_root, mode)):
                mode = "gemini" if mode == "local" else "local"  # fall back to what's trained
        artifact_dir = os.path.join(artifact_root, mode)
        cal_path = os.path.join(artifact_dir, "calibration.json")
        if not os.path.exists(cal_path):
            raise FileNotFoundError(
                f"no trained artifacts at {artifact_dir!r}. Train this configuration "
                f"first, e.g. `python -m mao.train --encoder <mode> "
                f"--node-encoder <node>` (EmbeddingGemma needs `ollama pull "
                f"embeddinggemma` running locally).")
        with open(cal_path) as f:
            cal = json.load(f)
        encoder = get_language_encoder(dim=cal["text_dim"], mode=cal["mode"])
        # reconstruct the node-feature encoder the model was trained with
        # (defaults to lexical hashing for artifacts saved before semantic nodes)
        node_encoder = get_node_encoder(mode=cal.get("node_mode", "hash"),
                                        dim=cal.get("node_dim", 64))
        model = JointModel(text_dim=cal["text_dim"], node_encoder=node_encoder)
        model.load_state_dict(torch.load(os.path.join(artifact_dir, "joint_model.pt"),
                                         weights_only=True))
        model.eval()
        bank_path = os.path.join(artifact_dir, "negative_bank.pt")
        bank = torch.load(bank_path, weights_only=True) if os.path.exists(bank_path) else None
        return cls(model, encoder, cal["tau_lo"], cal["tau_hi"], bank,
                   use_local_llm=use_local_llm)

    # ------------------------------------------------------------------

    def adjudicate(self, packet: HandoffPacket, proposed_action: str,
                   proposed_trace: str) -> Decision:
        graph = packet.graph
        t_vec = torch.tensor(self.encoder.encode(proposed_trace), dtype=torch.float32)

        with torch.no_grad():
            zt = self.model.embed_trace(t_vec)
            zg = torch.tensor(packet.embedding)
            alignment = float((zg * zt).sum())
            nearest_viol = float((self.negative_bank @ zt).max()) \
                if self.negative_bank is not None else 0.0

        if alignment >= self.tau_hi:
            verdict = Verdict.APPROVE
        elif alignment >= self.tau_lo:
            verdict = Verdict.FLAG_TO_HUMAN
        else:
            verdict = Verdict.REQUEST_REPLAN

        # attach the specific violated constraint from the graph snapshot
        violations = graph.check_action(proposed_action) \
            if verdict != Verdict.APPROVE else []

        # request-replan requires a structural witness: if the learned gate
        # fired but no violated edge can be named from the snapshot, defer to
        # a human rather than overruling the agent on suspicion alone
        if verdict == Verdict.REQUEST_REPLAN and not violations:
            verdict = Verdict.FLAG_TO_HUMAN

        message = self._message(verdict, proposed_action, violations, alignment)
        return Decision(verdict=verdict, alignment=alignment,
                        tau_lo=self.tau_lo, tau_hi=self.tau_hi,
                        nearest_violation_sim=nearest_viol,
                        violations=violations, message=message)

    def _message(self, verdict: Verdict, action: str,
                 violations: list[Violation], alignment: float) -> str:
        if verdict == Verdict.APPROVE:
            return f"'{action}' is structurally consistent with the task graph (alignment {alignment:.3f})."
        detail = violations[0].detail if violations else "alignment below threshold"
        template = {
            Verdict.FLAG_TO_HUMAN:
                f"'{action}' lands in the ambiguous band (alignment {alignment:.3f}); "
                f"deferring to human review. Possible issue: {detail}.",
            Verdict.REQUEST_REPLAN:
                f"'{action}' rejected (alignment {alignment:.3f}). Violated constraint: "
                f"{detail}. Re-plan with this dependency satisfied first.",
        }[verdict]

        if self.gemma and self.gemma.model:
            prompt = (
                "You are a local orchestration safety adjudicator. In one or two "
                f"sentences, explain to the acting agent why the action '{action}' was "
                f"given verdict '{verdict.value}' and what to do next. "
                f"Ground truth: {detail}. Be specific and imperative.")
            llm_msg = self.gemma.generate(prompt)
            if llm_msg:
                return llm_msg
        return template
