# AGENTS.md — llm-nano-vm (core)

Deterministic FSM runtime for LLM pipelines. This file is for coding agents; keep it short.

## Commands
- `tools/gate.sh` is the source of truth (import smoke, ruff check, mypy, pytest). Quote its
  first line (`commit: <sha> (<N> dirty)`) with any "N/N GREEN" claim.
- Never run `ruff --fix`. Never run `git commit`, `git push`, `git tag`, `git reset`.
- Verify imports/installs from a neutral directory (`cd ~`), not from the repo root.

## Hard rules
- No `eval()`/`exec()`; conditions go through `ASTEngine`. Python 3.10+ (`X | Y`), Pydantic v2,
  mypy 0 errors. `LiteLLMAdapter` stays a lazy import; `litellm<1.98.0` stays pinned.
- `_execute_loop` is recursive and has two entry points (`run`, `resume_with_program`): any
  change must hold on both and in every recursion frame.
- ASTEngine: no method calls, no arithmetic; parentheses group boolean sub-expressions only
  (max nesting 32); unknown characters raise
  `ASTEvalError`; precedence `not > and > or`; `a contains b` == `b in a`.
- Every change ships with a test that is RED on the old code. Never weaken, delete or skip an
  existing test to get green. Never change production code in a task that says "tests only".

## How to work (mandatory protocol — you are a small model, follow it literally)
1. Read the task brief in `.briefs/` fully. Do exactly one numbered step at a time.
2. Before using ANY class/function/attribute name, `grep` the source to confirm it exists and
   read its definition. Never invent APIs, signatures or file paths.
3. One file per step. Do not rewrite whole files; make small anchored edits.
4. After each step run the narrowest test, then paste the RAW command and RAW output.
5. If something is unclear, a command fails twice, or you are rate-limited: STOP and report
   what you know and what you do not know. Do not guess.
6. Finish with `tools/gate.sh`, `git diff --stat`, and a report that separates CONFIRMED
   (you ran it, output shown) from NOT CONFIRMED. Do not claim CI status; you cannot see it.
