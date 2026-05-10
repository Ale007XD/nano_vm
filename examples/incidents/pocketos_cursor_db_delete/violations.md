# Violations: LLM-initiated Production DB Deletion

## I1: LLM ∉ δ(S,E)→S'  
**Status:** VIOLATED  
**Evidence:** Step `select_targets` produced `target_databases` including `prod_main` which was not in the whitelist. The LLM's output directly determined the state transition (`execute_deletion` tool call) without an intermediate deterministic guard.

## I7: State whitelist enforcement  
**Status:** VIOLATED  
**Evidence:** Whitelist `allowed_databases` = ["test_alpha", "test_beta"]. LLM added `prod_main` and `analytics`. No validation before tool execution.

## I5: Compensation only via Saga  
**Status:** VIOLATED  
**Evidence:** Destructive action `DROP DATABASE` executed with no prior compensation step. No Saga transaction opened.

## I4: PII ∉ LLM context  
**Status:** LIKELY  
**Evidence:** Database names (`prod_main`, etc.) were exposed to LLM context, which may contain sensitive schema information.
