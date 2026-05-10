"""
tests.test_v070_sprint2_projection
===================================
Sprint 2: ProjectionLayer — тесты только nano_vm (без nano_vm_mcp).

Покрытие:
  - LLM target: PII regex, sensitive fields, CapabilityRef, tombstone
  - TRACE target: full data, tombstone sentinel, no PII redaction
  - TOOL target: capability filter, без policy → passthrough
  - DeterministicSanitizer: extra patterns, extra prefixes
  - project() convenience function
  - AbstractProjectionLayer ABC
  - Детерминизм: одинаковый input → одинаковый output
  - Вложенные dict/list в state
  - Интеграция ProjectionLayer + FSM lifecycle (StateContext)
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from nano_vm.models import CapabilityRef, PolicySnapshot, StateContext
from nano_vm.projection import (
    _PII_SENTINEL,
    _TOMBSTONE_SENTINEL,
    AbstractProjectionLayer,
    DeterministicSanitizer,
    ProjectionTarget,
    project,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def policy() -> PolicySnapshot:
    return PolicySnapshot(
        policy_id="p-test",
        version="1.0.0",
        tool_capabilities={
            "send_email": ["email.read_raw", "email.send"],
            "get_weather": ["weather.read"],
            "save_report": ["report.write"],
        },
    )


@pytest.fixture
def sanitizer() -> DeterministicSanitizer:
    return DeterministicSanitizer()


@pytest.fixture
def state_with_pii() -> StateContext:
    return StateContext(
        data={
            "user_email": "alice@example.com",
            "message": "Please contact bob@test.org about this",
            "score": 42,
            "safe_field": "hello world",
            "password": "s3cr3t",
            "token": "bearer-abc123",
        },
        step_outputs={
            "classify": "urgent",
            "summary": "Short summary without PII",
        },
    )


@pytest.fixture
def state_with_capref() -> StateContext:
    ref_live = CapabilityRef(ref_id="vault://users/42/email", salt="fixed-salt")
    ref_tomb = CapabilityRef(
        ref_id="vault://users/99/ssn", salt="tomb-salt", is_tombstone=True
    )
    return StateContext(
        data={
            "email_ref": ref_live,
            "ssn_ref": ref_tomb,
            "plain": "no-pii",
        },
        step_outputs={},
    )


# ---------------------------------------------------------------------------
# LLM target
# ---------------------------------------------------------------------------


class TestLLMProjection:
    def test_email_redacted(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.LLM)
        assert result["user_email"] == _PII_SENTINEL

    def test_email_in_string_redacted(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.LLM)
        assert "bob@test.org" not in result["message"]
        assert _PII_SENTINEL in result["message"]

    def test_non_pii_string_preserved(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.LLM)
        assert result["safe_field"] == "hello world"

    def test_numeric_preserved(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.LLM)
        assert result["score"] == 42

    def test_sensitive_field_prefix_redacted(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.LLM)
        assert result["password"] == _PII_SENTINEL
        assert result["token"] == _PII_SENTINEL

    def test_step_outputs_included(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.LLM)
        assert "__step_outputs__" in result
        assert result["__step_outputs__"]["classify"] == "urgent"

    def test_webhook_field_redacted(self, sanitizer: DeterministicSanitizer) -> None:
        state = StateContext(
            data={"__webhook__": {"status": "confirmed"}, "ok": "yes"}
        )
        result = sanitizer.project(state, ProjectionTarget.LLM)
        assert result["__webhook__"] == _PII_SENTINEL

    def test_capability_ref_live_hashed(
        self, sanitizer: DeterministicSanitizer, state_with_capref: StateContext
    ) -> None:
        result = sanitizer.project(state_with_capref, ProjectionTarget.LLM)
        ref = CapabilityRef(ref_id="vault://users/42/email", salt="fixed-salt")
        assert result["email_ref"] == ref.secure_hash()
        assert result["email_ref"] != "vault://users/42/email"

    def test_capability_ref_tombstone(
        self, sanitizer: DeterministicSanitizer, state_with_capref: StateContext
    ) -> None:
        result = sanitizer.project(state_with_capref, ProjectionTarget.LLM)
        assert result["ssn_ref"] == _TOMBSTONE_SENTINEL

    def test_nested_dict_sanitized(self, sanitizer: DeterministicSanitizer) -> None:
        state = StateContext(
            data={"nested": {"inner_email": "x@y.com", "safe": "ok"}}
        )
        result = sanitizer.project(state, ProjectionTarget.LLM)
        assert result["nested"]["inner_email"] == _PII_SENTINEL
        assert result["nested"]["safe"] == "ok"

    def test_list_sanitized(self, sanitizer: DeterministicSanitizer) -> None:
        state = StateContext(data={"items": ["a@b.com", "safe", 42]})
        result = sanitizer.project(state, ProjectionTarget.LLM)
        assert result["items"][0] == _PII_SENTINEL
        assert result["items"][1] == "safe"
        assert result["items"][2] == 42


# ---------------------------------------------------------------------------
# TRACE target
# ---------------------------------------------------------------------------


class TestTraceProjection:
    def test_email_not_redacted(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.TRACE)
        assert result["user_email"] == "alice@example.com"

    def test_sensitive_field_not_redacted(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.TRACE)
        assert result["password"] == "s3cr3t"

    def test_tombstone_sentinel_in_trace(
        self, sanitizer: DeterministicSanitizer, state_with_capref: StateContext
    ) -> None:
        result = sanitizer.project(state_with_capref, ProjectionTarget.TRACE)
        assert result["ssn_ref"] == _TOMBSTONE_SENTINEL

    def test_live_capref_hashed_in_trace(
        self, sanitizer: DeterministicSanitizer, state_with_capref: StateContext
    ) -> None:
        result = sanitizer.project(state_with_capref, ProjectionTarget.TRACE)
        ref = CapabilityRef(ref_id="vault://users/42/email", salt="fixed-salt")
        assert result["email_ref"] == ref.secure_hash()

    def test_step_outputs_included(
        self, sanitizer: DeterministicSanitizer, state_with_pii: StateContext
    ) -> None:
        result = sanitizer.project(state_with_pii, ProjectionTarget.TRACE)
        assert "__step_outputs__" in result

    def test_webhook_included_in_trace(self, sanitizer: DeterministicSanitizer) -> None:
        state = StateContext(data={"__webhook__": {"status": "confirmed"}})
        result = sanitizer.project(state, ProjectionTarget.TRACE)
        assert result["__webhook__"] == {"status": "confirmed"}


# ---------------------------------------------------------------------------
# TOOL target
# ---------------------------------------------------------------------------


class TestToolProjection:
    def test_only_capability_keys_returned(
        self, sanitizer: DeterministicSanitizer, policy: PolicySnapshot
    ) -> None:
        state = StateContext(
            data={
                "email.read_raw": "raw content",
                "email.send": True,
                "unrelated": "secret",
            }
        )
        result = sanitizer.project(
            state, ProjectionTarget.TOOL, policy=policy, tool_name="send_email"
        )
        assert "email.read_raw" in result
        assert "email.send" in result
        assert "unrelated" not in result

    def test_no_policy_returns_all_data(
        self, sanitizer: DeterministicSanitizer
    ) -> None:
        state = StateContext(data={"a": 1, "b": 2})
        result = sanitizer.project(state, ProjectionTarget.TOOL, policy=None)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_no_tool_name_returns_all_data(
        self, sanitizer: DeterministicSanitizer, policy: PolicySnapshot
    ) -> None:
        state = StateContext(data={"x": "y"})
        result = sanitizer.project(
            state, ProjectionTarget.TOOL, policy=policy, tool_name=None
        )
        assert result["x"] == "y"

    def test_tool_not_in_policy_returns_empty(
        self, sanitizer: DeterministicSanitizer, policy: PolicySnapshot
    ) -> None:
        state = StateContext(data={"a": 1})
        result = sanitizer.project(
            state, ProjectionTarget.TOOL, policy=policy, tool_name="unknown_tool"
        )
        assert result == {}

    def test_tombstone_in_tool_output(
        self, sanitizer: DeterministicSanitizer, policy: PolicySnapshot
    ) -> None:
        ref_tomb = CapabilityRef(ref_id="vault://x", salt="s", is_tombstone=True)
        state = StateContext(data={"weather.read": ref_tomb})
        result = sanitizer.project(
            state, ProjectionTarget.TOOL, policy=policy, tool_name="get_weather"
        )
        assert result["weather.read"] == _TOMBSTONE_SENTINEL

    def test_alias_project_for_tool(
        self, sanitizer: DeterministicSanitizer, policy: PolicySnapshot
    ) -> None:
        state = StateContext(data={"report.write": "data"})
        result = sanitizer.project_for_tool(state, "save_report", policy=policy)
        assert result["report.write"] == "data"


# ---------------------------------------------------------------------------
# DeterministicSanitizer — extra patterns/prefixes
# ---------------------------------------------------------------------------


class TestSanitizerExtensions:
    def test_extra_pii_pattern(self) -> None:
        pattern = (re.compile(r"\bACCT-\d+\b"), "[ACCOUNT_REDACTED]")
        sanitizer = DeterministicSanitizer(extra_pii_patterns=[pattern])
        state = StateContext(data={"info": "Account ACCT-12345 is active"})
        result = sanitizer.project(state, ProjectionTarget.LLM)
        assert "ACCT-12345" not in result["info"]
        assert "[ACCOUNT_REDACTED]" in result["info"]

    def test_extra_sensitive_prefix(self) -> None:
        sanitizer = DeterministicSanitizer(extra_sensitive_prefixes=("internal_",))
        state = StateContext(
            data={"internal_key": "classified", "public": "open"}
        )
        result = sanitizer.project(state, ProjectionTarget.LLM)
        assert result["internal_key"] == _PII_SENTINEL
        assert result["public"] == "open"


# ---------------------------------------------------------------------------
# project() convenience function
# ---------------------------------------------------------------------------


class TestProjectConvenience:
    def test_project_llm(self) -> None:
        state = StateContext(data={"email": "x@y.com"})
        result = project(state, ProjectionTarget.LLM)
        assert result["email"] == _PII_SENTINEL

    def test_project_trace(self) -> None:
        state = StateContext(data={"email": "x@y.com"})
        result = project(state, ProjectionTarget.TRACE)
        assert result["email"] == "x@y.com"

    def test_project_tool(self, policy: PolicySnapshot) -> None:
        state = StateContext(
            data={"weather.read": "sunny", "other": "secret"}
        )
        result = project(
            state, ProjectionTarget.TOOL, policy=policy, tool_name="get_weather"
        )
        assert "weather.read" in result
        assert "other" not in result

    def test_determinism(self) -> None:
        state = StateContext(
            data={"email": "a@b.com", "score": 10},
            step_outputs={"s1": "out1"},
        )
        r1 = project(state, ProjectionTarget.LLM)
        r2 = project(state, ProjectionTarget.LLM)
        assert r1 == r2


# ---------------------------------------------------------------------------
# AbstractProjectionLayer ABC
# ---------------------------------------------------------------------------


class TestAbstractProjectionLayer:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AbstractProjectionLayer()  # type: ignore[abstract]

    def test_custom_implementation(self) -> None:
        class PassthroughLayer(AbstractProjectionLayer):
            def project(
                self,
                state: StateContext,
                target: ProjectionTarget,
                policy: PolicySnapshot | None = None,
                tool_name: str | None = None,
            ) -> dict[str, Any]:
                return dict(state.data)

        layer = PassthroughLayer()
        state = StateContext(data={"k": "v"})
        assert layer.project(state, ProjectionTarget.LLM) == {"k": "v"}
        assert layer.project_for_llm(state) == {"k": "v"}
        assert layer.project_for_trace(state) == {"k": "v"}
        assert layer.project_for_tool(state, "any_tool") == {"k": "v"}


# ---------------------------------------------------------------------------
# Интеграция: ProjectionLayer + FSM lifecycle (StateContext)
# ---------------------------------------------------------------------------


class TestProjectionFSMIntegration:
    def test_project_after_step_output(
        self, sanitizer: DeterministicSanitizer
    ) -> None:
        state = StateContext(
            data={"user_input": "hello"},
            step_outputs={"classify": "urgent", "summary": "ok"},
        )
        result = sanitizer.project_for_llm(state)
        assert result["__step_outputs__"]["classify"] == "urgent"
        assert result["__step_outputs__"]["summary"] == "ok"

    def test_tombstone_propagation_in_llm(
        self, sanitizer: DeterministicSanitizer
    ) -> None:
        dead_ref = CapabilityRef(ref_id="vault://x", salt="s", is_tombstone=True)
        state = StateContext(data={}, step_outputs={"extract": dead_ref})
        result = sanitizer.project_for_llm(state)
        assert result["__step_outputs__"]["extract"] == _TOMBSTONE_SENTINEL

    def test_state_immutability(self, sanitizer: DeterministicSanitizer) -> None:
        state = StateContext(data={"email": "x@y.com", "score": 5})
        original_data = dict(state.data)
        sanitizer.project(state, ProjectionTarget.LLM)
        assert dict(state.data) == original_data

    def test_empty_state(self, sanitizer: DeterministicSanitizer) -> None:
        state = StateContext()
        for target in ProjectionTarget:
            result = sanitizer.project(state, target)
            assert isinstance(result, dict)
