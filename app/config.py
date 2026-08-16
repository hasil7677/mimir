import logging
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ServerConfig(BaseModel):
    port: int = 8080
    host: str = "127.0.0.1"
    api_key: str = ""
    cors_origins: list[str] = Field(default_factory=list)


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379"
    hot_ttl_hours: int = 24
    cache_ttl_hours: int = 6
    freq_ttl_days: int = 30


class QdrantConfig(BaseModel):
    mode: str = "local"  # "local" (embedded) | "server"
    path: str = "~/.mimir/qdrant"
    collection_name: str = "l1_memories"


class KuzuConfig(BaseModel):
    path: str = "~/.mimir/kuzu"


class DuckdbConfig(BaseModel):
    path: str = "~/.mimir/memories.db"


class VaultConfig(BaseModel):
    """The OKF vault: the agent's memory as an Obsidian-compatible directory
    of markdown files. This is the ground-truth human-readable layer — the
    databases are indexes over it, not the other way around."""

    path: str = "~/.mimir/vault"


class StorageConfig(BaseModel):
    backend: str = "full"  # "full" (Redis+Qdrant+KuZu+DuckDB) | "lite" (SQLite+sqlite-vec)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    kuzu: KuzuConfig = Field(default_factory=KuzuConfig)
    duckdb: DuckdbConfig = Field(default_factory=DuckdbConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)


class EmbeddingConfig(BaseModel):
    # Defaults to "none" deliberately: out of the box, with no config file and
    # no key, this must make zero outbound network calls. "openai" only
    # activates once someone explicitly sets a key (or points base_url at a
    # local Ollama instance); "fastembed" only activates once someone
    # explicitly opts in too — its first use downloads model weights (a
    # one-time, anonymous file fetch, no user content sent), so it's a
    # deliberate choice, never a silent default that phones home either.
    provider: str = "none"  # "openai" | "local" | "fastembed" | "none"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536

    # fastembed only: runs fully in-process (ONNX), no server, no API key.
    # Dense model powers semantic search; sparse model is Qdrant's own BM25
    # implementation, which is what makes Qdrant's native hybrid fusion
    # possible instead of hand-rolling our own BM25 + RRF merge.
    fastembed_dense_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_sparse_model: str = "Qdrant/bm25"
    fastembed_cache_dir: str = "~/.mimir/models"


class LlmConfig(BaseModel):
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    extraction_model: str = ""
    timeout_ms: int = 30000
    # Gemini 2.5 models think by default, and on the Vertex API those thinking
    # tokens are charged against maxOutputTokens — the same budget the actual
    # answer has to fit in. Measured on a real 60k-char session: 1918 of a
    # 2000-token budget went to thinking, leaving 78 tokens, which truncated
    # the extraction JSON after a single fact object. Same session with
    # thinking off: 44 facts. Every task this engine asks an LLM to do is
    # structured enumeration, not reasoning, so 0 is the right default.
    #   0  = disabled (default)
    #   >0 = explicit token budget for thinking
    #   <0 = omit the field entirely and let the model decide
    thinking_budget: int = 0


class PipelineConfig(BaseModel):
    l1_every_n_turns: int = 5
    l1_idle_timeout_seconds: int = 600
    l2_every_n_memories: int = 20
    l3_every_n_memories: int = 50
    enable_warmup: bool = True


class RecallWeights(BaseModel):
    semantic: float = 0.45
    frequency: float = 0.20
    recency: float = 0.25
    graph: float = 0.10


class RecallConfig(BaseModel):
    strategy: str = "hybrid"  # "hybrid" | "vector" | "keyword" | "graph"
    # Was 5. Verified live against a 33-fact synthetic user (PersonaMem eval):
    # the fact a query needed was ranked #6-9, present at max_results=10,
    # absent at 5. 5 is fine for a light user; once real usage accumulates
    # enough facts to have this problem at all, cutting to 5 costs more
    # accuracy than the extra context costs in token budget.
    max_results: int = 10
    recall_threshold: float = 0.30
    cache_threshold: float = 0.92
    # Bumped in step with max_results — otherwise the extra candidates just
    # get discarded by the char-budget trim (_assemble pops weakest-first)
    # instead of actually reaching the reader.
    max_context_chars: int = 6000
    weights: RecallWeights = Field(default_factory=RecallWeights)
    decay_rate: float = 0.05


class ExtractionConfig(BaseModel):
    # Was 20, which never bound on anything: a truncation bug meant sessions
    # were landing a single fact each, so the cap was unreachable. With that
    # fixed, real 32k-token sessions extract 26-32 facts and 20 silently
    # discarded the tail — and the tail is ordered by nothing in particular,
    # so it was not dropping the least useful ones.
    max_memories_per_session: int = 50
    min_priority: int = 50
    enable_dedup: bool = True
    enable_entity_extraction: bool = True


class CaptureConfig(BaseModel):
    l0_retention_days: int = 0
    exclude_agents: list[str] = Field(default_factory=list)


class PiiConfig(BaseModel):
    """Pro feature — parsed either way so config validation doesn't
    depend on which tier is running; enforcement is gated elsewhere."""

    enabled: bool = False
    types: list[str] = Field(default_factory=lambda: ["EMAIL", "PHONE", "SSN", "CREDIT_CARD"])
    strategy: str = "mask"


class ComplianceConfig(BaseModel):
    audit_log_enabled: bool = False
    data_residency: str = "local"


class MimirConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    recall: RecallConfig = Field(default_factory=RecallConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    pii: PiiConfig = Field(default_factory=PiiConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)


def _substitute_env_vars(raw_yaml: str) -> str:
    """Replaces ${VAR_NAME} with the environment variable's value (empty
    string if unset) — keeps secrets out of the committed yaml file."""
    return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), raw_yaml)


# Direct environment overrides for the handful of knobs that get swept rather
# than configured. `${VAR}` substitution already covers secrets, but it only
# works for values a yaml file already mentions — so varying one parameter
# across runs meant writing a whole config file per run, or reaching in and
# mutating `settings` after import (which is what this project's own eval
# scripts were reduced to doing). These apply with or without a config file,
# and always win over it: an explicit env var is a deliberate act.
_ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "MIMIR_DECAY_RATE": ("recall", "decay_rate", float),
    "MIMIR_MAX_RESULTS": ("recall", "max_results", int),
    "MIMIR_RECALL_THRESHOLD": ("recall", "recall_threshold", float),
    "MIMIR_MAX_CONTEXT_CHARS": ("recall", "max_context_chars", int),
    "MIMIR_MIN_PRIORITY": ("extraction", "min_priority", int),
    "MIMIR_THINKING_BUDGET": ("llm", "thinking_budget", int),
}


def _apply_env_overrides(config: MimirConfig) -> MimirConfig:
    for env_var, (section, field, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            setattr(getattr(config, section), field, caster(raw))
        except (ValueError, TypeError):
            # A malformed override is a typo in a sweep script, not a reason to
            # refuse to boot — the documented default still applies.
            logger.warning("ignoring %s=%r — not a valid %s", env_var, raw, caster.__name__)
    return config


def load_config(path: str | None = None) -> MimirConfig:
    """Local-first by design: if no config file is found, every setting falls
    back to its documented default and the gateway still boots — a config
    file is an override, never a requirement.
    """
    config_path = Path(path or os.environ.get("MIMIR_CONFIG", "mimir.yaml"))
    if not config_path.exists():
        return _apply_env_overrides(MimirConfig())

    raw = config_path.read_text()
    substituted = _substitute_env_vars(raw)
    data = yaml.safe_load(substituted) or {}
    return _apply_env_overrides(MimirConfig(**data))


settings = load_config()
