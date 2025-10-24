# security/frontier_filter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pickle

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "Please install sentence-transformers: pip install sentence-transformers"
    ) from e


@dataclass
class FrontierFilter:
    model_name: str
    benign_centroid: np.ndarray
    attack_centroid: np.ndarray
    threshold: float = 0.0
    _embedder: SentenceTransformer = None  # runtime only (not pickled)

    def __post_init__(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs

    @staticmethod
    def _cos(v: np.ndarray, c: np.ndarray) -> float:
        return float(np.dot(v, c))

    def score(self, text: str) -> Tuple[float, float, float]:
        v = self.embed([text])[0]
        sb = self._cos(v, self.benign_centroid)
        sa = self._cos(v, self.attack_centroid)
        return sb, sa, (sb - sa)

    def is_attack(self, text: str) -> Tuple[bool, Dict[str, float]]:
        sb, sa, margin = self.score(text)
        blocked = margin < self.threshold
        return blocked, {
            "sim_benign": sb,
            "sim_attack": sa,
            "margin": margin,
            "threshold": self.threshold,
        }

    @classmethod
    def fit(
        cls,
        benign_texts: List[str],
        attack_texts: List[str],
        model_name: str = "BAAI/bge-small-en-v1.5",
        threshold_percentile: float = 5.0,
    ) -> "FrontierFilter":
        tmp = cls(model_name=model_name, benign_centroid=np.zeros(1), attack_centroid=np.zeros(1))
        B = tmp.embed(benign_texts)
        A = tmp.embed(attack_texts)
        bc = B.mean(axis=0); bc /= np.linalg.norm(bc) + 1e-12
        ac = A.mean(axis=0); ac /= np.linalg.norm(ac) + 1e-12
        margins_b = (B @ bc) - (B @ ac)
        thr = float(np.percentile(margins_b, threshold_percentile))
        tmp.benign_centroid, tmp.attack_centroid, tmp.threshold = bc, ac, thr
        return tmp

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "benign_centroid": self.benign_centroid,
                    "attack_centroid": self.attack_centroid,
                    "threshold": self.threshold,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FrontierFilter":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(**d)
        return obj
