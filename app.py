import streamlit as st
import json
import os
import time
from langchain_core.messages import HumanMessage
from graph_pipeline import build_semagc_graph
from baseline_pipeline import build_baseline_graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SemaGC Runtime",
    page_icon="🧠",
    layout="wide"
)
# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 SemaGC: Semantic Garbage Collection Runtime")
st.markdown("""
**SemaGC** is a context management system for multi-agent LLM pipelines.
It uses semantic similarity to prune irrelevant messages and compress
relevant ones into a dense Micro-Rationale — reducing state size without
losing task-critical information.
""")
st.divider()

# ── Task input ────────────────────────────────────────────────────────────────
st.subheader("📋 Task Configuration")

task = st.text_area(
    label="Enter a complex multi-agent task:",
    value="Build a complete Python REST API using FastAPI that includes user authentication with JWT tokens, a PostgreSQL database connection, CRUD operations for a todo list, input validation, error handling, rate limiting, and full test coverage",
    height=100
)

col_config1, col_config2, col_config3 = st.columns(3)

with col_config1:
    token_threshold = st.slider(
        "GC Token Threshold",
        min_value=4,
        max_value=20,
        value=12,
        help="GC triggers when message count exceeds this number"
    )

with col_config2:
    drift_threshold = st.slider(
        "Drift Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05,
        help="Messages with similarity below this are pruned"
    )

with col_config3:
    preserve_n = st.slider(
        "Preserve Last N Messages",
        min_value=1,
        max_value=5,
        value=2,
        help="Always keep the N most recent messages"
    )

# ── Sync UI config to SemaGC runtime ─────────────────────────────────────────
def update_semagc_config(threshold, drift, preserve):
    import semagc_runtime
    semagc_runtime.SEMAGC_CONFIG["token_threshold"] = threshold
    semagc_runtime.SEMAGC_CONFIG["drift_threshold"] = drift
    semagc_runtime.SEMAGC_CONFIG["preserve_last_n"] = preserve

st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
if st.button("🚀 Run Orchestration Comparison", type="primary", use_container_width=True):

    # Sync config from sliders
    update_semagc_config(token_threshold, drift_threshold, preserve_n)

    # Clear old metrics log
    if os.path.exists("results/metrics.jsonl"):
        os.remove("results/metrics.jsonl")

    # ── Run Baseline ──────────────────────────────────────────────────────────
    st.subheader("⏳ Running pipelines...")
    progress = st.progress(0, text="Starting baseline pipeline...")

    baseline_graph = build_baseline_graph()
    baseline_initial = {
        "memory_vault":      [HumanMessage(content=task)],
        "execution_metrics": {},
        "current_task":      task,
        "task_result":       None,
    }

    progress.progress(10, text="Running baseline (no GC)...")
    baseline_start = time.time()
    baseline_result = baseline_graph.invoke(
        baseline_initial,
        {"configurable": {"thread_id": f"baseline-{int(time.time())}"}}
    )
    baseline_time = round(time.time() - baseline_start, 2)
    progress.progress(50, text="Baseline done. Running SemaGC pipeline...")

    # ── Run SemaGC ────────────────────────────────────────────────────────────
    semagc_graph = build_semagc_graph()
    semagc_initial = {
        "memory_vault":      [HumanMessage(content=task)],
        "execution_metrics": {},
        "current_task":      task,
        "task_result":       None,
    }

    semagc_start = time.time()
    semagc_result = semagc_graph.invoke(
        semagc_initial,
        {"configurable": {"thread_id": f"semagc-{int(time.time())}"}}
    )
    semagc_time = round(time.time() - semagc_start, 2)
    progress.progress(100, text="Both pipelines complete!")
    time.sleep(0.5)
    progress.empty()

# ── Metrics row ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Results")

    baseline_count = len(baseline_result["memory_vault"])
    semagc_count   = len(semagc_result["memory_vault"])
    reduction      = round((1 - semagc_count / baseline_count) * 100) if baseline_count > 0 else 0
    time_diff      = round(baseline_time - semagc_time, 2)

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        label="📦 Baseline Messages",
        value=baseline_count
    )
    m2.metric(
        label="🗜 SemaGC Messages",
        value=semagc_count,
        delta=f"-{baseline_count - semagc_count}" if baseline_count > semagc_count else "0",
        delta_color="inverse"
    )
    m3.metric(
        label="📉 State Reduction",
        value=f"{reduction}%"
    )
    m4.metric(
        label="⏱ Baseline Time",
        value=f"{baseline_time}s"
    )
    m5.metric(
        label="⚡ SemaGC Time",
        value=f"{semagc_time}s",
        delta=f"{time_diff}s" if time_diff != 0 else "0s",
        delta_color="inverse"
    )
# ── Side by side vault view ───────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Memory Vault Comparison")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### 📚 Baseline — Unmanaged State")
        st.caption(f"{baseline_count} messages — grows unchecked")
        for i, msg in enumerate(baseline_result["memory_vault"]):
            label = "👤 User Goal" if msg.type == "human" else f"🤖 Agent Message {i}"
            with st.expander(label, expanded=(i == 0)):
                st.text(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

    with right_col:
        st.markdown("### 🧠 SemaGC — Managed State")
        st.caption(f"{semagc_count} messages — GC optimized")
        for i, msg in enumerate(semagc_result["memory_vault"]):
            if msg.type == "human":
                label = "👤 User Goal"
            elif "SemaGC Micro-Rationale" in msg.content:
                label = "🗜 Micro-Rationale (compressed)"
            else:
                label = f"🤖 Agent Message {i}"
            with st.expander(label, expanded=(i == 0)):
                st.text(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)
# ── Final results ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("✅ Final Task Output")

    res_left, res_right = st.columns(2)

    with res_left:
        st.markdown("**Baseline Output**")
        st.code(baseline_result["task_result"] or "No result", language="markdown")

    with res_right:
        st.markdown("**SemaGC Output**")
        st.code(semagc_result["task_result"] or "No result", language="markdown")

    # ── GC event log ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ SemaGC Collection Event Log")

    if os.path.exists("results/metrics.jsonl"):
        gc_events = []
        with open("results/metrics.jsonl") as f:
            for line in f:
                event = json.loads(line)
                if event.get("event") == "gc_collection":
                    gc_events.append(event)

        if gc_events:
            for event in gc_events:
                st.success(
                    f"⚡ GC triggered | "
                    f"Before: {event['before_count']} msgs → "
                    f"After: {event['after_count']} msgs | "
                    f"Pruned: {event['pruned_count']} | "
                    f"Compressed: {event['compressed_count']} → 1 Micro-Rationale | "
                    f"Reduction: {event['reduction_pct']}%"
                )
        else:
            st.info(
                "ℹ️ GC did not trigger this run. "
                f"Vault stayed at {semagc_count} messages "
                f"(threshold is {token_threshold}). "
                "Lower the GC Token Threshold slider to force GC, "
                "or run a more complex multi-step task."
            )
    else:
        st.warning("No metrics file found.")

    # ── All metrics log ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Full Raw Metrics Log"):
        if os.path.exists("results/metrics.jsonl"):
            with open("results/metrics.jsonl") as f:
                for line in f:
                    event = json.loads(line)
                    st.json(event)

# ── Placeholder shown before first run ───────────────────────────────────────
else:
    st.info(
        "👆 Configure your task above and click **Run Orchestration Comparison** to start. "
        "Both pipelines will run and results will appear here."
    )

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Baseline Pipeline**")
        st.markdown("Runs 4 agents with no memory management. State grows with every message.")
    with col2:
        st.markdown("**2. SemaGC Pipeline**")
        st.markdown("Same 4 agents but with the GC reducer. Prunes and compresses automatically.")
    with col3:
        st.markdown("**3. Compare Results**")
        st.markdown("See message counts, reduction %, timing, and the full GC event log.")
