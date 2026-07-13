import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_registry import (
    get_guardrail_initializer_from_hooks,
    InMemoryGuardrailHandler,
)
from litellm.types.guardrails import GuardrailEventHooks, Guardrail, LitellmParams


def test_get_guardrail_initializer_from_hooks():
    initializers = get_guardrail_initializer_from_hooks()
    assert "aim" in initializers


def test_guardrail_class_registry():
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    assert "aim" in guardrail_class_registry
    assert "aporia" in guardrail_class_registry


def test_noma_registry_resolution():
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaGuardrail
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 import NomaV2Guardrail
    from litellm.proxy.guardrails.guardrail_registry import (
        guardrail_class_registry,
        guardrail_initializer_registry,
    )

    assert guardrail_class_registry["noma"] is NomaGuardrail
    assert guardrail_class_registry["noma_v2"] is NomaV2Guardrail
    assert "noma" in guardrail_initializer_registry
    assert "noma_v2" in guardrail_initializer_registry


def test_update_in_memory_guardrail():
    handler = InMemoryGuardrailHandler()
    handler.guardrail_id_to_custom_guardrail["123"] = CustomGuardrail(
        guardrail_name="test-guardrail",
        default_on=False,
        event_hook=GuardrailEventHooks.pre_call,
    )

    handler.update_in_memory_guardrail(
        "123",
        Guardrail(
            guardrail_name="test-guardrail",
            litellm_params=LitellmParams(
                guardrail="test-guardrail", mode="pre_call", default_on=True
            ),
        ),
    )

    assert (
        handler.guardrail_id_to_custom_guardrail["123"].should_run_guardrail(
            data={}, event_type=GuardrailEventHooks.pre_call
        )
        is True
    )
    assert (
        handler.guardrail_id_to_custom_guardrail["123"].event_hook
        is GuardrailEventHooks.pre_call
    )


def _make_guardrail(guardrail_id: str, name: str = "g") -> Guardrail:
    return Guardrail(
        guardrail_id=guardrail_id,
        guardrail_name=name,
        litellm_params=LitellmParams(guardrail=name, mode="pre_call", default_on=False),
    )


def test_reconcile_db_guardrails_drops_stale_db_entries_only():
    """
    The reconcile pass must drop in-memory entries marked source='db' that are
    missing from the DB result, and never touch source='config' entries.
    Models the multi-pod case where another pod deleted a DB-backed guardrail.
    """
    handler = InMemoryGuardrailHandler()

    # Two DB-backed entries on this pod (synced from earlier polling cycles)
    handler.IN_MEMORY_GUARDRAILS["db-keep"] = _make_guardrail("db-keep")
    handler.IN_MEMORY_GUARDRAILS["db-stale"] = _make_guardrail("db-stale")
    handler._sources["db-keep"] = "db"
    handler._sources["db-stale"] = "db"

    # One config-loaded entry that must survive reconciliation
    handler.IN_MEMORY_GUARDRAILS["cfg"] = _make_guardrail("cfg")
    handler._sources["cfg"] = "config"

    # The DB now only contains db-keep — db-stale was deleted on another pod.
    removed = handler.reconcile_db_guardrails(db_guardrail_ids={"db-keep"})

    assert removed == ["db-stale"]
    assert "db-stale" not in handler.IN_MEMORY_GUARDRAILS
    assert "db-stale" not in handler._sources
    assert "db-keep" in handler.IN_MEMORY_GUARDRAILS
    assert "cfg" in handler.IN_MEMORY_GUARDRAILS
    assert handler._sources["cfg"] == "config"


def test_reconcile_does_not_drop_config_entries_missing_from_db():
    """A config-only guardrail (no DB row) must never be reconciled away."""
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["cfg-only"] = _make_guardrail("cfg-only")
    handler._sources["cfg-only"] = "config"

    removed = handler.reconcile_db_guardrails(db_guardrail_ids=set())

    assert removed == []
    assert "cfg-only" in handler.IN_MEMORY_GUARDRAILS


def test_get_source_returns_marker_set_at_insert():
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["a"] = _make_guardrail("a")
    handler._sources["a"] = "db"
    handler.IN_MEMORY_GUARDRAILS["b"] = _make_guardrail("b")
    handler._sources["b"] = "config"

    assert handler.get_source("a") == "db"
    assert handler.get_source("b") == "config"
    assert handler.get_source("missing") is None


def test_delete_in_memory_guardrail_clears_source_marker():
    handler = InMemoryGuardrailHandler()
    handler.IN_MEMORY_GUARDRAILS["a"] = _make_guardrail("a")
    handler._sources["a"] = "db"

    handler.delete_in_memory_guardrail("a")

    assert "a" not in handler.IN_MEMORY_GUARDRAILS
    assert "a" not in handler._sources
    assert handler.get_source("a") is None


def test_initialize_guardrail_early_return_updates_source_marker():
    """
    When initialize_guardrail is called for a guardrail that already exists
    in memory, the early-return path must still honor the caller's source.
    Otherwise a racing polling tick that placed a DB entry in memory first
    would leave a later config-init call wrongly marked as 'db' (or vice
    versa), and the entry would be reconciled with the wrong classification.
    """
    handler = InMemoryGuardrailHandler()
    # Simulate a polling tick already placing the entry as DB-backed.
    handler.IN_MEMORY_GUARDRAILS["collide"] = _make_guardrail("collide", name="bedrock")
    handler._sources["collide"] = "db"

    # Config init re-visits the same id (e.g., hot-reload, or UUID collision).
    g = Guardrail(
        guardrail_id="collide",
        guardrail_name="bedrock",
        litellm_params=LitellmParams(
            guardrail="bedrock", mode="pre_call", default_on=False
        ),
    )
    handler.initialize_guardrail(guardrail=g, source="config")

    assert handler.get_source("collide") == "config"

    # And the symmetric direction: db sync should override an entry left
    # marked as 'config' from a stale init path.
    handler.initialize_guardrail(guardrail=g, source="db")
    assert handler.get_source("collide") == "db"


def test_sync_guardrail_from_db_marks_source_db_when_unchanged():
    """
    sync_guardrail_from_db must enforce source='db' even when params are
    unchanged, so a config entry whose UUID happens to collide with a later
    DB row gets re-tagged correctly.
    """
    handler = InMemoryGuardrailHandler()
    g = _make_guardrail("collide")
    handler.IN_MEMORY_GUARDRAILS["collide"] = g
    handler._sources["collide"] = "config"

    handler.sync_guardrail_from_db(g)

    assert handler.get_source("collide") == "db"


def test_reinitialize_guardrail_leaves_old_guardrail_intact_on_build_failure():
    """
    A failed rebuild (e.g. transient error, unsupported/misconfigured
    guardrail type) must never tear down an already-working guardrail. Build
    the replacement first; only swap in the new one once it succeeds.
    """
    handler = InMemoryGuardrailHandler()
    working = _make_guardrail("gid", name="bedrock")
    working_callback = CustomGuardrail(
        guardrail_name="bedrock",
        default_on=False,
        event_hook=GuardrailEventHooks.pre_call,
    )
    handler.IN_MEMORY_GUARDRAILS["gid"] = working
    handler.guardrail_id_to_custom_guardrail["gid"] = working_callback
    handler._sources["gid"] = "db"

    broken = Guardrail(
        guardrail_id="gid",
        guardrail_name="bedrock",
        litellm_params=LitellmParams(
            guardrail="not-a-real-guardrail-type", mode="pre_call", default_on=False
        ),
    )

    with pytest.raises(ValueError):
        handler.reinitialize_guardrail(guardrail=broken, source="db")

    assert handler.IN_MEMORY_GUARDRAILS["gid"] is working
    assert handler.guardrail_id_to_custom_guardrail["gid"] is working_callback
    assert handler._sources["gid"] == "db"


def test_resolve_env_secret_value_passthrough_and_errors(monkeypatch):
    assert (
        InMemoryGuardrailHandler._resolve_env_secret_value("literal-value", "api_key")
        == "literal-value"
    )
    assert (
        InMemoryGuardrailHandler._resolve_env_secret_value(None, "api_key") is None
    )

    monkeypatch.setenv("TEST_GUARDRAIL_SECRET", "resolved-value")
    assert (
        InMemoryGuardrailHandler._resolve_env_secret_value(
            "os.environ/TEST_GUARDRAIL_SECRET", "api_key"
        )
        == "resolved-value"
    )

    monkeypatch.delenv("TEST_GUARDRAIL_UNSET_SECRET", raising=False)
    with pytest.raises(ValueError):
        InMemoryGuardrailHandler._resolve_env_secret_value(
            "os.environ/TEST_GUARDRAIL_UNSET_SECRET", "api_key"
        )


def test_has_guardrail_params_changed_stable_for_resolved_env_secret(monkeypatch):
    """
    The in-memory copy holds the resolved secret value (post-build), while a
    freshly-fetched DB row always holds the raw os.environ/ reference. These
    must compare as unchanged so a guardrail configured via env var
    references isn't torn down and rebuilt on every DB polling tick.
    """
    monkeypatch.setenv("TEST_GUARDRAIL_SECRET", "resolved-value")
    handler = InMemoryGuardrailHandler()

    existing = Guardrail(
        guardrail_id="gid",
        guardrail_name="bedrock",
        litellm_params=LitellmParams(
            guardrail="bedrock",
            mode="pre_call",
            default_on=False,
            api_key="resolved-value",
        ),
    )
    handler.IN_MEMORY_GUARDRAILS["gid"] = existing

    from_db_same_secret = Guardrail(
        guardrail_id="gid",
        guardrail_name="bedrock",
        litellm_params=LitellmParams(
            guardrail="bedrock",
            mode="pre_call",
            default_on=False,
            api_key="os.environ/TEST_GUARDRAIL_SECRET",
        ),
    )
    assert (
        handler._has_guardrail_params_changed("gid", from_db_same_secret) is False
    )

    from_db_different_secret = Guardrail(
        guardrail_id="gid",
        guardrail_name="bedrock",
        litellm_params=LitellmParams(
            guardrail="bedrock",
            mode="pre_call",
            default_on=False,
            api_key="a-genuinely-different-value",
        ),
    )
    assert (
        handler._has_guardrail_params_changed("gid", from_db_different_secret) is True
    )

    monkeypatch.delenv("TEST_GUARDRAIL_SECRET", raising=False)
    assert (
        handler._has_guardrail_params_changed("gid", from_db_same_secret) is False
    )
