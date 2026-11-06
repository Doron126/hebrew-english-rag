import os
import logging
from pathlib import Path

try:
    from .config_env import DEFAULT_EMBED_MODEL, embedding_model_kwargs, load_project_environment
except ImportError:
    from config_env import DEFAULT_EMBED_MODEL, embedding_model_kwargs, load_project_environment

load_project_environment(Path(__file__).resolve().parent)

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _assert_model_device(model, requested: str) -> str:
    actual = str(getattr(model, "device", "unknown")).casefold()
    if requested.casefold() == "cpu" and actual != "unknown" and not actual.startswith("cpu"):
        raise RuntimeError(f"Embedder requested CPU but initialized on {actual}")
    logger.info("Embedding runtime device requested=%s actual=%s", requested, actual)
    return actual

class Embedder:
    def __init__(self, name: str | None = None):
        name = name or os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
        device = os.getenv("EMBED_DEVICE", "cpu").strip() or "cpu"
        self.model = SentenceTransformer(
            name, device=device, model_kwargs=embedding_model_kwargs(device),
        )
        self.device = _assert_model_device(self.model, device)
    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
