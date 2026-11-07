# CLAUDE.md — llm-nano-vm (core)

Deterministic FSM runtime for LLM pipelines. Public repo: keep this file free of
internal process references.

## Commands (source of truth: `tools/gate.sh`)
- `tools/gate.sh` — import smoke, ruff (check only), mypy, pytest. A claim of "N/N GREEN"
  must quote its first line (`commit: <sha> (<N> dirty)`).
- Never run `ruff --fix` in CI; formatting changes go in their own commit.
- Verify installs/imports from a neutral directory (`cd ~`), never from the repo root:
  the CWD shadows site-packages.

## Hard rules
- No `eval()`/`exec()`; conditions go through `ASTEngine` only.
- Python 3.10+ (`X | Y` unions), Pydantic v2, `mypy` 0 errors.
- `adapters/__init__.py` keeps `LiteLLMAdapter` a lazy import (PEP 562); bare
  `pip install llm-nano-vm` must not need litellm. `litellm<1.98.0` pin stays.
- `_execute_loop` is recursive (CONDITION -> CONDITION) and has two entry points
  (`run`, `resume_with_program`): any new handler/pre-flight must be applied on
  both entry points and in every recursion frame.
- `ASTEngine` limits: no method calls, no arithmetic; parentheses group boolean
  sub-expressions only (max nesting 32); unknown
  characters raise `ASTEvalError`; precedence `not > and > or`;
  `a contains b` == `b in a`.
- Every deliverable ships with tests. A bug fix needs a test that is RED on the old code;
  prove each guard with a negative control (revert the fix, exactly that test must fail).
- Public text (README, CHANGELOG, comments) never names internal process files.

## Working style
- Read the source before editing; prefer small anchored edits over rewrites.
- Do not run `git commit`, `git push` or tag; stop after the gate and show `git diff --stat`.
- Release order (for the maintainer): gate -> push -> CI on THAT commit (all jobs) ->
  CHANGELOG fold -> push -> CI -> tag -> publish -> smoke from a neutral directory.
