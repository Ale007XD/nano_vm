import os
import time
import random
import copy
from dataclasses import dataclass
from typing import List, Dict, Any

# =========================
# CONFIG (inputs per run)
# =========================
N_RUNS = 1000
MODEL = "gpt-4o-mini"

MAX_RETRIES = 2
FAIL_PROB = 0.3
FRAUD_PROB = 0.2
ELIGIBLE_PROB = 0.8


# =========================
# StepResult
# =========================
@dataclass
class StepResult:
    step: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    state_before: str
    state_after: str


# =========================
# Token estimator
# =========================
def estimate_tokens(x):
    return max(1, len(str(x)) // 4)


# =========================
# LLM (real + fallback)
# =========================
def llm_decide(context):
    tokens = estimate_tokens(context)

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            start = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": str(context)}],
                temperature=0.7,
            )
            latency = time.time() - start

            text = resp.choices[0].message.content.lower()
            return ("retry" in text), tokens, latency

        except Exception:
            pass

    # fallback
    start = time.time()
    val = random.random() > 0.5
    latency = time.time() - start

    return val, tokens, latency


# =========================
# Fake API with failures
# =========================
class PaymentAPI:
    def __init__(self):
        self.refund_count = 0

    def refund(self):
        if random.random() < FAIL_PROB:
            return {"status": "fail"}

        self.refund_count += 1
        return {"status": "success", "id": f"r{self.refund_count}"}


# =========================
# RAW AGENT
# =========================
def run_raw():
    api = PaymentAPI()
    tokens = 0
    t0 = time.time()

    eligible = random.random() < ELIGIBLE_PROB
    fraud = random.random() < FRAUD_PROB

    if not eligible or fraud:
        return {"refunds": 0, "tokens": 0, "time": 0}

    res = api.refund()

    retry, tk, lat = llm_decide(res)
    tokens += tk

    retries = 0
    while retry and retries < MAX_RETRIES:
        res = api.refund()
        retry, tk, lat2 = llm_decide(res)
        tokens += tk
        retries += 1

    return {
        "refunds": api.refund_count,
        "tokens": tokens,
        "time": time.time() - t0
    }


# =========================
# FSM Runtime
# =========================
class Runtime:
    def __init__(self):
        self.state = "INIT"
        self.trace: List[StepResult] = []
        self.api = PaymentAPI()

    def step(self, name, fn, inp):
        before = self.state
        out = fn(inp)
        self.state = out.get("next_state", self.state)

        self.trace.append(StepResult(
            step=name,
            input=copy.deepcopy(inp),
            output=copy.deepcopy(out),
            state_before=before,
            state_after=self.state
        ))

        return out


def safe_refund(rt: Runtime):
    # invariant: max 1 success
    for s in rt.trace:
        if s.step.startswith("refund") and s.output.get("api", {}).get("status") == "success":
            return {"blocked": True, "next_state": rt.state}

    res = rt.api.refund()
    return {"api": res, "next_state": "REFUNDED"}


def run_fsm():
    rt = Runtime()
    tokens = 0
    t0 = time.time()

    eligible = random.random() < ELIGIBLE_PROB
    fraud = random.random() < FRAUD_PROB

    if not eligible or fraud:
        return {"refunds": 0, "tokens": 0, "time": 0}

    r1 = rt.step("refund", lambda _: safe_refund(rt), {})

    retry, tk, lat = llm_decide(r1)
    tokens += tk

    retries = 0

    while retry and retries < MAX_RETRIES:
        r = rt.step("refund_retry", lambda _: safe_refund(rt), {})
        retry, tk, lat2 = llm_decide(r)
        tokens += tk
        retries += 1

    return {
        "refunds": rt.api.refund_count,
        "tokens": tokens,
        "time": time.time() - t0
    }


# =========================
# Benchmark
# =========================
def run_benchmark():
    raw_errors = 0
    fsm_errors = 0

    raw_tokens = 0
    fsm_tokens = 0

    raw_time = 0
    fsm_time = 0

    for _ in range(N_RUNS):
        random.seed()

        raw = run_raw()
        fsm = run_fsm()

        if raw["refunds"] > 1:
            raw_errors += 1

        if fsm["refunds"] > 1:
            fsm_errors += 1

        raw_tokens += raw["tokens"]
        fsm_tokens += fsm["tokens"]

        raw_time += raw["time"]
        fsm_time += fsm["time"]

    return {
        "raw_errors": raw_errors,
        "fsm_errors": fsm_errors,
        "raw_tokens": raw_tokens,
        "fsm_tokens": fsm_tokens,
        "raw_time": raw_time,
        "fsm_time": fsm_time
    }


# =========================
# TABLE
# =========================
def print_table(res):
    print("\n=== CONFIG ===\n")

    config_rows = [
        ["Runs", N_RUNS],
        ["Model", MODEL],
        ["Max retries", MAX_RETRIES],
        ["Fail prob", FAIL_PROB],
        ["Fraud prob", FRAUD_PROB],
        ["Eligible prob", ELIGIBLE_PROB],
    ]

    for k, v in config_rows:
        print(f"{k:<20} {v}")

    print("\n=== RESULTS ===\n")

    headers = ["Metric", "Raw", "FSM"]
    rows = [
        ["Double refunds", res["raw_errors"], res["fsm_errors"]],
        ["Total tokens", res["raw_tokens"], res["fsm_tokens"]],
        ["Avg tokens/run", res["raw_tokens"]//N_RUNS, res["fsm_tokens"]//N_RUNS],
        ["Total time (s)", round(res["raw_time"], 3), round(res["fsm_time"], 3)],
        ["Avg time/run", round(res["raw_time"]/N_RUNS, 5), round(res["fsm_time"]/N_RUNS, 5)],
    ]

    print(f"{headers[0]:<20}{headers[1]:<15}{headers[2]:<15}")
    print("-" * 50)

    for r in rows:
        print(f"{str(r[0]):<20}{str(r[1]):<15}{str(r[2]):<15}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    results = run_benchmark()
    print_table(results)
