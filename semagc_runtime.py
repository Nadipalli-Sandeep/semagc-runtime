import numpy as np
import os
import json
import time
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Model initialisation ─────────────────────────────────────────────────────
embeddings_engine = SentenceTransformer("all-MiniLM-L6-v2")
compression_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ── Runtime configuration ────────────────────────────────────────────────────
SEMAGC_CONFIG = {
    "token_threshold":  6,
    "drift_threshold":  0.35,
    "preserve_last_n":  2,
}

# ── Metrics collector ────────────────────────────────────────────────────────
def log_metrics(event: dict):
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.jsonl", "a") as f:
        f.write(json.dumps({**event, "timestamp": time.time()}) + "\n")

# ── Semantic Drift computation ────────────────────────────────────────────────
def compute_semantic_drift(goal_text: str, candidate_text: str) -> float:
    if not candidate_text or len(candidate_text.strip()) < 10:
        return 0.0
    vec_goal = embeddings_engine.encode(goal_text, convert_to_numpy=True)
    vec_cand = embeddings_engine.encode(candidate_text, convert_to_numpy=True)
    similarity = np.dot(vec_goal, vec_cand) / (
        np.linalg.norm(vec_goal) * np.linalg.norm(vec_cand) + 1e-9
    )
    return float(similarity)

# ── Context Volatility check ─────────────────────────────────────────────────
def compute_context_volatility(messages: List[BaseMessage]) -> int:
    return len(messages)

# ── Micro-Rationale generator ─────────────────────────────────────────────────
def generate_micro_rationale(messages_to_compress: List[str]) -> str:
    if not messages_to_compress:
        return ""
    time.sleep(2)
    joined = "\n\n".join(messages_to_compress)[-2000:]
    prompt = (
        "You are a memory compaction engine. Condense the following agent "
        "execution logs into a dense, factual bullet list. "
        "Remove ALL pleasantries, apologies, repeated phrases, and raw code dumps. "
        "Keep only final values, key decisions, and confirmed facts.\n\n"
        f"{joined}"
    )
    response = compression_llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ── The SemaGC Garbage Collector ─────────────────────────────────────────────
def semagc_garbage_collector(
    existing: List[BaseMessage],
    new_updates: List[BaseMessage]
) -> List[BaseMessage]:

    combined = existing + new_updates
    volatility = compute_context_volatility(combined)

    log_metrics({
        "event":         "state_update",
        "message_count": volatility,
        "gc_triggered":  False
    })

    # ── Check if GC should trigger ────────────────────────────────────────────
    if volatility <= SEMAGC_CONFIG["token_threshold"]:
        return combined

    # ── GC TRIGGERED ─────────────────────────────────────────────────────────
    print(f"\n⚡ SemaGC daemon active — {volatility} messages. Running collection...")

    goal_message = combined[0]
    goal_text    = goal_message.content

    preserve_n  = SEMAGC_CONFIG["preserve_last_n"]
    middle_msgs = combined[1:-preserve_n] if preserve_n > 0 else combined[1:]
    tail_msgs   = combined[-preserve_n:] if preserve_n > 0 else []

    # ── Semantic Pruner ───────────────────────────────────────────────────────
    to_compress  = []
    pruned_count = 0

    for msg in middle_msgs:
        content = msg.content if hasattr(msg, "content") else str(msg)
        drift   = compute_semantic_drift(goal_text, content)

        if drift < SEMAGC_CONFIG["drift_threshold"]:
            pruned_count += 1
            print(f"   🗑  Pruned: drift={drift:.3f} | {content[:60]}...")
        else:
            to_compress.append(f"[{msg.type}]: {content}")

    # ── Context Defragmenter ──────────────────────────────────────────────────
    micro_rationale = ""
    if to_compress:
        micro_rationale = generate_micro_rationale(to_compress)

    # ── Rebuild compressed state ──────────────────────────────────────────────
    new_state = [goal_message]
    if micro_rationale:
        new_state.append(AIMessage(
            content=f"[SemaGC Micro-Rationale — {len(to_compress)} steps compressed]:\n{micro_rationale}"
        ))
    new_state.extend(tail_msgs)

    log_metrics({
        "event":            "gc_collection",
        "before_count":     volatility,
        "after_count":      len(new_state),
        "pruned_count":     pruned_count,
        "compressed_count": len(to_compress),
        "reduction_pct":    round((1 - len(new_state) / volatility) * 100, 1)
    })

    print(f"   ✅ Compacted: {volatility} → {len(new_state)} messages "
          f"({pruned_count} pruned, {len(to_compress)} compressed)")

    return new_state

# ── The SemaGC State Schema ───────────────────────────────────────────────────
class SemaGCState(TypedDict):
    memory_vault:      Annotated[List[BaseMessage], semagc_garbage_collector]
    execution_metrics: dict
    current_task:      str
    task_result:       Optional[str]