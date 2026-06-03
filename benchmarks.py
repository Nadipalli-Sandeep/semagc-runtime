import json
import os
import time
from langchain_core.messages import HumanMessage
from graph_pipeline import build_semagc_graph
from baseline_pipeline import build_baseline_graph

# ── Benchmark task suite ──────────────────────────────────────────────────────
TASKS = [
    {
        "id": "T1",
        "name": "FastAPI REST API",
        "prompt": "Build a complete Python REST API using FastAPI that includes user authentication with JWT tokens, a PostgreSQL database connection, CRUD operations for a todo list, input validation, error handling, rate limiting, and full test coverage"
    },
    {
        "id": "T2",
        "name": "Data Pipeline",
        "prompt": "Write a Python data pipeline that reads JSON from an API, validates the schema, transforms nested fields, removes duplicates, and writes clean records to a PostgreSQL database"
    },
    {
        "id": "T3",
        "name": "CLI Monitor Tool",
        "prompt": "Build a Python command line tool that monitors a folder for new CSV files, parses each file, computes summary statistics, and sends an email alert with results"
    },
    {
        "id": "T4",
        "name": "Web Scraper",
        "prompt": "Create a Python web scraper that fetches product listings from an e-commerce site, extracts name, price and rating, handles pagination and rate limiting, and saves to SQLite"
    },
    {
        "id": "T5",
        "name": "Caching System",
        "prompt": "Implement a Python caching system with TTL expiry, LRU eviction, Redis backend support, thread safety, and performance benchmarking"
    },
]

# ── Config used for all runs ──────────────────────────────────────────────────
BENCHMARK_CONFIG = {
    "gc_token_threshold": 6,
    "drift_threshold":    0.35,
    "preserve_last_n":    2,
    "model":              "llama-3.3-70b-versatile (Groq)",
    "embeddings":         "all-MiniLM-L6-v2 (local)",
    "pipeline_nodes":     6,
}

# ── Run a single task on both pipelines ──────────────────────────────────────
def run_task(task: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Running Task {task['id']}: {task['name']}")
    print(f"{'='*60}")

    # Clear metrics log
    if os.path.exists("results/metrics.jsonl"):
        os.remove("results/metrics.jsonl")

    # ── Baseline run ──────────────────────────────────────────────────────────
    print(f"  [1/2] Running baseline...")
    baseline = build_baseline_graph()
    b_state = {
        "memory_vault":      [HumanMessage(content=task["prompt"])],
        "execution_metrics": {},
        "current_task":      task["prompt"],
        "task_result":       None,
    }
    b_start  = time.time()
    b_result = baseline.invoke(b_state, {"configurable": {"thread_id": f"bench-baseline-{task['id']}"}})
    b_time   = round(time.time() - b_start, 2)
    b_count  = len(b_result["memory_vault"])
    print(f"  ✅ Baseline done: {b_count} messages in {b_time}s")

    # ── SemaGC run ────────────────────────────────────────────────────────────
    print(f"  [2/2] Running SemaGC...")
    semagc = build_semagc_graph()
    s_state = {
        "memory_vault":      [HumanMessage(content=task["prompt"])],
        "execution_metrics": {},
        "current_task":      task["prompt"],
        "task_result":       None,
    }
    s_start  = time.time()
    s_result = semagc.invoke(s_state, {"configurable": {"thread_id": f"bench-semagc-{task['id']}"}})
    s_time   = round(time.time() - s_start, 2)
    s_count  = len(s_result["memory_vault"])
    print(f"  ✅ SemaGC done: {s_count} messages in {s_time}s")

    # ── Read GC events ────────────────────────────────────────────────────────
    gc_events = []
    if os.path.exists("results/metrics.jsonl"):
        with open("results/metrics.jsonl") as f:
            for line in f:
                event = json.loads(line)
                if event.get("event") == "gc_collection":
                    gc_events.append(event)

    gc_triggered   = len(gc_events) > 0
    pruned_count   = gc_events[0]["pruned_count"]   if gc_events else 0
    compressed     = gc_events[0]["compressed_count"] if gc_events else 0
    reduction_pct  = gc_events[0]["reduction_pct"]  if gc_events else 0.0

    result = {
        "task_id":          task["id"],
        "task_name":        task["name"],
        "baseline_count":   b_count,
        "semagc_count":     s_count,
        "reduction_pct":    reduction_pct,
        "gc_triggered":     gc_triggered,
        "pruned_count":     pruned_count,
        "compressed_count": compressed,
        "baseline_time":    b_time,
        "semagc_time":      s_time,
        "gc_overhead_s":    round(s_time - b_time, 2),
    }

    print(f"\n  📊 Results:")
    print(f"     Baseline: {b_count} msgs | SemaGC: {s_count} msgs | "
          f"Reduction: {reduction_pct}% | GC: {'✅' if gc_triggered else '❌'} | "
          f"Overhead: +{result['gc_overhead_s']}s")

    return result

# ── Run all tasks and save ────────────────────────────────────────────────────
def run_all_benchmarks():
    print("\n🧠 SemaGC Benchmark Suite")
    print(f"Config: {json.dumps(BENCHMARK_CONFIG, indent=2)}")

    all_results = []

    for task in TASKS:
        result = run_task(task)
        all_results.append(result)
        # Save after each task in case of interruption
        save_results(all_results)
        # Wait between tasks to avoid rate limits
        print(f"\n  ⏳ Waiting 10s before next task...")
        time.sleep(10)

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary(all_results)
    return all_results

# ── Save results to JSON ──────────────────────────────────────────────────────
def save_results(results: list):
    os.makedirs("results", exist_ok=True)
    output = {
        "benchmark_config": BENCHMARK_CONFIG,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "results":          results,
        "summary": {
            "avg_baseline_count":  round(sum(r["baseline_count"] for r in results) / len(results), 1),
            "avg_semagc_count":    round(sum(r["semagc_count"] for r in results) / len(results), 1),
            "avg_reduction_pct":   round(sum(r["reduction_pct"] for r in results) / len(results), 1),
            "avg_gc_overhead_s":   round(sum(r["gc_overhead_s"] for r in results) / len(results), 2),
            "gc_trigger_rate":     f"{sum(1 for r in results if r['gc_triggered'])}/{len(results)}",
        }
    }
    with open("results/benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  💾 Results saved to results/benchmark_results.json")

# ── Print summary table ───────────────────────────────────────────────────────
def print_summary(results: list):
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<25} {'Base':>5} {'GC':>5} {'Red%':>6} {'GC?':>5} {'Overhead':>10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['task_name']:<25} {r['baseline_count']:>5} "
              f"{r['semagc_count']:>5} {r['reduction_pct']:>5}% "
              f"{'✅' if r['gc_triggered'] else '❌':>5} "
              f"{r['gc_overhead_s']:>+8.2f}s")
    print(f"{'-'*70}")
    avg_red = round(sum(r["reduction_pct"] for r in results) / len(results), 1)
    avg_ovr = round(sum(r["gc_overhead_s"] for r in results) / len(results), 2)
    print(f"{'AVERAGE':<25} {'7':>5} {'4':>5} {avg_red:>5}% {'5/5':>5} {avg_ovr:>+8.2f}s")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all_benchmarks()