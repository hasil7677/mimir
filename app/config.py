import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

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
    provider: str = "openai"  # "openai" | "local" | "none"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536


class LlmConfig(BaseModel):
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    extraction_model: str = ""
    timeout_ms: int = 30000


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
    max_results: int = 5
    recall_threshold: float = 0.30
    cache_threshold: float = 0.92
    max_context_chars: int = 4000
    weights: RecallWeights = Field(default_factory=RecallWeights)
    decay_rate: float = 0.05


class ExtractionConfig(BaseModel):
    max_memories_per_session: int = 20
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


def load_config(path: str | None = None) -> MimirConfig:
    """Local-first by design: if no config file is found, every setting falls
    back to its documented default and the gateway still boots — a config
    file is an override, never a requirement.
    """
    config_path = Path(path or os.environ.get("MIMIR_CONFIG", "mimir.yaml"))
    if not config_path.exists():
        return MimirConfig()

    raw = config_path.read_text()
    substituted = _substitute_env_vars(raw)
    data = yaml.safe_load(substituted) or {}
    return MimirConfig(**data)


settings = load_config()
