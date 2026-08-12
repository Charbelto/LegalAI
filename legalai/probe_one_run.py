"""Send a single benchmark-shaped request and report what came back.

    python probe_one_run.py [mode]

A diagnostic, not part of the experiment: it writes nothing to any runs file. Its
purpose is to answer, before committing to a multi-day benchmark, three questions
that only a real request can answer - do the expert models actually load under
this interpreter, how long does one run take, and is the answer a real answer
rather than the abstention sentence.
"""

import json
import sys
import time
import uuid

import requests

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"

dataset = json.load(open("eval_dataset.json", encoding="utf-8"))
item = dataset[0]

# A UNIQUE session id per probe. A fixed one silently invalidated three
# comparisons: the memory agent loads persisted history for a session, so each
# probe saw the previous probe's answer as a prior assistant turn and echoed it -
# making a freshly retrained adapter reproduce the OLD adapter's output almost
# verbatim and look unchanged. benchmark.py already uses a unique id per
# (arm, mode, query, repeat) for exactly this reason; the diagnostic did not.
session_id = f"probe_{MODE}_{uuid.uuid4().hex[:8]}"

print(f"mode={MODE}  session={session_id}  query={item['query'][:80]}")
started = time.perf_counter()
response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={
        "message": item["query"],
        "session_id": session_id,
        "fetch_news": False,
        "expert_execution_mode": MODE,
        "seed": 1000,
    },
    timeout=5400,
)
elapsed = time.perf_counter() - started

print(f"status={response.status_code}  elapsed={elapsed:.1f}s")
if response.status_code != 200:
    print("BODY:", response.text[:800])
    sys.exit(1)

data = response.json()
print(f"experts_run={data.get('experts_run')}  abstained={data.get('abstained')}")
print(f"abstained_experts={data.get('abstained_experts')}")
print(f"tokens prompt/completion = {data.get('prompt_tokens')}/{data.get('completion_tokens')}")
print(f"truncation_warnings={data.get('truncation_warnings')}")
timings = data.get("agent_timings_ms", {})
if timings:
    ordered = sorted(timings.items(), key=lambda kv: -kv[1])[:6]
    print("slowest nodes (ms): " + ", ".join(f"{k}={v:.0f}" for k, v in ordered))
answer = data.get("response", "")
print(f"answer length: {len(answer)} chars, {len(answer.split())} words")
print("--- answer ---")
print(answer[:700])
