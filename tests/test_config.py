"""app.config.load_config: defaults without a file, yaml overrides on top,
and direct environment overrides on top of that.

The env-override layer exists because `${VAR}` substitution only reaches
values a yaml file already mentions, so sweeping one parameter across runs
otherwise meant writing a config file per run.
"""

from app.config import load_config


def test_defaults_apply_with_no_config_file(tmp_path):
    config = load_config(str(tmp_path / "does-not-exist.yaml"))

    assert config.recall.decay_rate == 0.05
    assert config.recall.max_results == 10
    assert config.llm.thinking_budget == 0
    assert config.extraction.max_memories_per_session == 50


def test_env_override_applies_without_a_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMIR_DECAY_RATE", "0.0")
    config = load_config(str(tmp_path / "does-not-exist.yaml"))

    assert config.recall.decay_rate == 0.0


def test_env_override_beats_the_config_file(tmp_path, monkeypatch):
    """An explicit env var is a deliberate act — it wins over the file."""
    config_file = tmp_path / "mimir.yaml"
    config_file.write_text("recall:\n  decay_rate: 0.05\n  max_results: 10\n", encoding="utf-8")
    monkeypatch.setenv("MIMIR_DECAY_RATE", "0.0")
    monkeypatch.setenv("MIMIR_MAX_RESULTS", "25")

    config = load_config(str(config_file))

    assert config.recall.decay_rate == 0.0
    assert config.recall.max_results == 25


def test_unset_and_empty_env_vars_leave_the_config_alone(tmp_path, monkeypatch):
    config_file = tmp_path / "mimir.yaml"
    config_file.write_text("recall:\n  decay_rate: 0.07\n", encoding="utf-8")
    monkeypatch.setenv("MIMIR_DECAY_RATE", "")  # e.g. `export MIMIR_DECAY_RATE=` in a script

    config = load_config(str(config_file))

    assert config.recall.decay_rate == 0.07


def test_malformed_override_is_ignored_rather_than_fatal(tmp_path, monkeypatch, caplog):
    """A typo in a sweep script shouldn't stop the engine from booting."""
    monkeypatch.setenv("MIMIR_MAX_RESULTS", "twenty")

    with caplog.at_level("WARNING"):
        config = load_config(str(tmp_path / "does-not-exist.yaml"))

    assert config.recall.max_results == 10  # documented default still stands
    assert "MIMIR_MAX_RESULTS" in caplog.text


def test_every_declared_override_reaches_its_field(tmp_path, monkeypatch):
    """Guards the mapping itself: a renamed config field would otherwise make
    an override silently stop working, which is the exact class of silent
    failure this module keeps running into."""
    monkeypatch.setenv("MIMIR_DECAY_RATE", "0.5")
    monkeypatch.setenv("MIMIR_MAX_RESULTS", "3")
    monkeypatch.setenv("MIMIR_RECALL_THRESHOLD", "0.42")
    monkeypatch.setenv("MIMIR_MAX_CONTEXT_CHARS", "999")
    monkeypatch.setenv("MIMIR_MIN_PRIORITY", "7")
    monkeypatch.setenv("MIMIR_THINKING_BUDGET", "-1")

    config = load_config(str(tmp_path / "does-not-exist.yaml"))

    assert config.recall.decay_rate == 0.5
    assert config.recall.max_results == 3
    assert config.recall.recall_threshold == 0.42
    assert config.recall.max_context_chars == 999
    assert config.extraction.min_priority == 7
    assert config.llm.thinking_budget == -1


def test_yaml_env_substitution_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_SECRET", "sk-from-env")
    config_file = tmp_path / "mimir.yaml"
    config_file.write_text('llm:\n  api_key: "${SOME_SECRET}"\n', encoding="utf-8")

    assert load_config(str(config_file)).llm.api_key == "sk-from-env"
