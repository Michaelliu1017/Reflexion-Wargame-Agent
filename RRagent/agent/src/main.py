"""
main.py

Main game loop with Strategic Plan lifecycle:
- Game start: load National Strategy → initialize Strategic Plans
- Each round: generate Round Plan (referencing plans) → execute phases → track progress
- Every 2 rounds: strategic reassessment (may add/modify/abandon plans)
- Game end: reflexion evaluates each plan → updates National Strategy
"""
import os
import time
import uuid
from pyBanner import banner, info, effect

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from agent import (
    run_full_turn, _summarize_round, _reassess_strategy,
    _initialize_strategic_plans, LLM_MODEL,
)
from bridge_client import TripleABridgeClient
from memory import (
    GameMemory, ReflexionEngine, StrategicPlan,
    load_national_strategy, board_snapshot,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_client = TripleABridgeClient()


def _get_state_with_retry(max_wait_seconds: int = 300):
    """
    get_state with long-wait retry.
    After Japan's turn, TripleA processes other AI turns and Bridge may be temporarily unresponsive.
    Retries every 10s until max_wait_seconds is exceeded, then returns None.
    """
    waited = 0
    interval = 10
    while waited <= max_wait_seconds:
        try:
            return _client.get_state()
        except (TimeoutError, OSError, Exception) as e:
            err_type = type(e).__name__
            print(
                f"  [Bridge] get_state timeout/failure ({err_type}), "
                f"waiting {interval}s before retry (waited {waited}s / {max_wait_seconds}s)..."
            )
            time.sleep(interval)
            waited += interval
    return None


# ─────────────────────────────────────────────────────────────
# 里程碑定义
# each milestone:
#   id         — unique identifier
#   name       — human-readable label
#   check      — function(state) -> bool, True = milestone reached
#   max_rounds — stop after this many rounds even if not reached
# ─────────────────────────────────────────────────────────────

def _owns(state: dict, territory: str) -> bool:
    """Check if Japanese own a specific territory."""
    for t in state.get("territories", []):
        if t.get("name") == territory:
            return t.get("owner", "") in ("Japanese", "Japan")
    return False


_CHINA_TERRITORIES = [
    "Manchuria", "Jehol", "Shantung", "Kiangsu", "Kiangsi", "Kwangtung",
    "Kwangsi", "Chahar", "Anhwe", "Hunan", "Yunnan", "Hopei",
    "Kweichow", "Szechwan", "Shensi", "Suiyuan", "Kansu", "Tsinghai", "Sikang",
]

def _china_army_value(state: dict) -> int:
    """Sum IPC value of remaining CHINESE units (excluding Japanese units on Chinese territory).

    unitsSummary includes ALL nations' units. unitsByTerritory contains only Japanese units.
    We subtract Japanese units from the total to isolate Chinese units.
    """
    _val = {"infantry": 3, "artillery": 4, "fighter": 10, "aaGun": 5}
    jp_units_by_terr = state.get("unitsByTerritory", {})
    total = 0
    for t in state.get("territories", []):
        owner = t.get("owner", "") or ""
        if owner == "Chinese":
            name = t.get("name", "")
            all_units = t.get("unitsSummary", {})
            jp_here = jp_units_by_terr.get(name, {})
            for unit_type, count in all_units.items():
                chinese_count = max(count - jp_here.get(unit_type, 0), 0)
                total += _val.get(unit_type, 0) * chinese_count
    return total


MILESTONES = [
    {
        "id": "m1",
        "name": "Conquer China (control ≥80% territories AND Chinese army ≤ 6 IPC)",
        "check": lambda state: (
            sum(1 for t in _CHINA_TERRITORIES if _owns(state, t))
            >= len(_CHINA_TERRITORIES) * 0.8
            and _china_army_value(state) <= 6
        ),
        "max_rounds": 6,
    },
]


# ─────────────────────────────────────────────────────────────
# 里程碑检查
# ─────────────────────────────────────────────────────────────

def check_milestone(state: dict, milestone: dict) -> bool:
    """Return True if the given milestone condition is met."""
    try:
        return milestone["check"](state)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# 主游戏循环
# ─────────────────────────────────────────────────────────────

_CONFIG_LABELS = {
    "A": "Baseline    — no reflection, no RAG",
    "B": "RAG         — game-level reflection + experience retrieval",
    "C": "Full        — reflexion + criticizer (round + game + RAG + criticizer)",
}


def run_game(milestone_id: str = "m1", config: str = "C"):
    """
    Run the game loop with configurable Reflexion ablation.

    Configurations (for ablation study):
      A — Baseline: No reflection, no RAG. Round Plan only.
      B — RAG:      Game-level reflection + experience retrieval.
      C — Full:     Round + game reflection, per-phase lessons, criticizer.
    """
    config = config.upper()
    if config not in _CONFIG_LABELS:
        print(f"[Error] Unknown config '{config}'. Use A, B, or C.")
        return None, []

    enable_rag        = config in ("B", "C")
    enable_reflexion  = config == "C"
    enable_criticizer = config == "C"

    milestone = next((m for m in MILESTONES if m["id"] == milestone_id), MILESTONES[0])
    print(f"\n{'=' * 60}")
    print(f"  Config:     [{config}] {_CONFIG_LABELS[config]}")
    print(f"  Objective:  {milestone['name']}")
    print(f"  Max rounds: {milestone['max_rounds']}")
    print(f"  ── Modules ──")
    print(f"    RAG experience retrieval : {'ON' if enable_rag else 'OFF'}")
    print(f"    Reflexion (round + game) : {'ON' if enable_reflexion else 'OFF'}")
    print(f"    Criticizer               : {'ON' if enable_criticizer else 'OFF'}")
    print(f"{'=' * 60}\n")

    # ── Set up memory, NS, and reflexion ──
    ns_path = os.path.join(PROJECT_ROOT, "knowledge", "national_strategy.json")
    exp_json = os.path.join(PROJECT_ROOT, "memory", "experiences.json")

    ns = load_national_strategy(ns_path)

    memory = GameMemory(
        rules_path=os.path.join(PROJECT_ROOT, "knowledge", "rules.txt"),
        rules_index_path=os.path.join(PROJECT_ROOT, "knowledge", "rules_index"),
        exp_index_path=os.path.join(PROJECT_ROOT, "knowledge", "exp_index"),
        experiences_json_path=exp_json,
    )
    reflexion = ReflexionEngine(
        memory,
        experiences_path=exp_json,
        ns_path=ns_path,
        reflect_model=LLM_MODEL,
    )

    game_id = str(uuid.uuid4())[:8]
    game_log: list[str] = []
    round_summaries: list[str] = []
    plan_tracker: list[StrategicPlan] = []
    plans_initialized = False
    current_round = 0
    prev_snapshot = None  # BoardSnapshot from end of previous round
    round_scores: list[float] = []  # score trajectory per round

    while current_round < milestone["max_rounds"]:
        current_round += 1
        print(f"\n[Round {current_round} / {milestone['max_rounds']}]")

        # Wait for our turn
        print("  Waiting for Japanese turn (Bridge must be running)...")
        dots = 0
        while not _client.is_our_turn():
            time.sleep(5)
            dots += 1
            if dots % 6 == 0:
                print("  Still waiting... ensure game + bridge are running on port 8081")

        current_phase = _client.get_phase()
        print(f"  Japanese turn detected! Starting at phase: {current_phase}")

        # ── Snapshot board BEFORE our turn ──
        try:
            pre_state = _client.get_state()
            pre_snap = board_snapshot(pre_state, current_round)
            print(f"  [Snapshot] Round {current_round} start — score: {pre_snap.score:.1f}, "
                  f"JP China IPC: {pre_snap.jp_china_ipc}, China army: {pre_snap.china_army_value}, "
                  f"Control: {pre_snap.china_controlled}/{pre_snap.china_total}")
        except Exception:
            pre_snap = None

        # ── Initialize Strategic Plans (once, before round 1) ──
        if not plans_initialized:
            plan_tracker = _initialize_strategic_plans(
                ns, round_num=current_round, use_criticizer=enable_criticizer,
            )
            plans_initialized = True

        # ── Layer 3: Strategic Reassessment every 2 rounds ──
        if current_round >= 2 and current_round % 2 == 0:
            plan_tracker = _reassess_strategy(
                round_num=current_round,
                plan_tracker=plan_tracker,
                round_summaries=round_summaries,
                use_criticizer=enable_criticizer,
            )

        # ── RAG retrieval (configs B, C) ──
        rag_context = ""
        if enable_rag:
            active_plans = [p for p in plan_tracker if p.status == "active"]
            active_plans.sort(key=lambda p: p.target_round)
            stage = "Early" if current_round <= 3 else ("Mid" if current_round <= 6 else "Late")
            if active_plans:
                top = active_plans[0]
                rag_query = f"Japan {stage} game round {current_round}: how to {top.name}"
            else:
                rag_query = f"Japan {stage} game round {current_round} strategy"
            try:
                rag_context = memory.retrieve(rag_query, k=3)
            except Exception:
                rag_context = ""

        # ── Run full turn ──
        turn_log = run_full_turn(
            rag_context=rag_context,
            start_phase=current_phase,
            round_num=current_round,
            plan_tracker=plan_tracker,
            prev_round_summaries=round_summaries,
            memory=memory if enable_reflexion else None,
            enable_criticizer=enable_criticizer,
        )
        game_log.extend(turn_log)

        # ── Safety check: retry if turn didn't fully complete ──
        # Skip safety retries on the final round to prevent accidental round 7 execution
        if current_round < milestone["max_rounds"]:
            for _retry in range(3):
                time.sleep(3)
                try:
                    if not _client.is_our_turn():
                        break
                    state_now = _client.get_state()
                    round_now = state_now.get("game", {}).get("round", current_round)
                    if round_now > current_round:
                        break
                    remaining = _client.get_phase()
                    print(
                        f"\n  [Safety Check] Japan turn incomplete (still in {remaining}, "
                        f"round={round_now}) — resuming (attempt {_retry + 1})..."
                    )
                    extra_log = run_full_turn(
                        rag_context=rag_context,
                        start_phase=remaining,
                        round_num=current_round,
                        plan_tracker=plan_tracker,
                        prev_round_summaries=round_summaries,
                        memory=memory if enable_reflexion else None,
                        enable_criticizer=enable_criticizer,
                    )
                    game_log.extend(extra_log)
                except Exception:
                    break
        else:
            # Final round: just wait briefly for game to settle
            time.sleep(5)

        # ── Post-turn snapshot + Round-level reflection ──
        _wait_s = 60 if current_round >= milestone["max_rounds"] else 300
        state = _get_state_with_retry(max_wait_seconds=_wait_s)
        if state is None:
            print("  [Warning] Could not get post-turn state (Bridge timeout) — skipping milestone check")
            continue

        post_snap = board_snapshot(state, current_round)
        round_scores.append(post_snap.score)
        print(f"  [Snapshot] Round {current_round} end — score: {post_snap.score:.1f}, "
              f"JP China IPC: {post_snap.jp_china_ipc}, China army: {post_snap.china_army_value}, "
              f"Control: {post_snap.china_controlled}/{post_snap.china_total}")

        # ── Round summary (with quantitative board diff) + plan progress ──
        compare_snap = prev_snapshot if prev_snapshot is not None else pre_snap
        summary = _summarize_round(current_round, turn_log, pre_snap, post_snap)
        round_summaries.append(summary)
        for p in plan_tracker:
            if p.status == "active":
                p.progress.append(f"Round {current_round}: {summary}")

        if enable_reflexion and compare_snap is not None:
            reflexion.reflect_round(
                prev_snap=compare_snap,
                curr_snap=post_snap,
                round_log=turn_log,
                plan_tracker=plan_tracker,
            )
        prev_snapshot = post_snap

        # ── Milestone check ──
        if check_milestone(state, milestone):
            print(f"\n  Milestone REACHED: {milestone['name']}")
            result = f"SUCCESS - {milestone['name']}"
            # Mark relevant plans as completed
            for p in plan_tracker:
                if p.status == "active":
                    p.status = "completed"
                    p.progress.append(f"Round {current_round}: MILESTONE ACHIEVED")
            break
    else:
        print(f"\n  Max rounds reached without achieving milestone.")
        result = f"TIMEOUT - {milestone['name']} not achieved in {milestone['max_rounds']} rounds"

    # ── Post-game Strategic Reflexion (configs B, C) ──
    reviews = []
    if enable_rag:
        print(f"\n  Running Strategic Reflexion for game {game_id} (round {current_round})...")
        reviews = reflexion.reflect_and_store(
            game_id=game_id,
            game_log=game_log,
            result=result,
            round_num=current_round,
            plan_tracker=plan_tracker,
        )
    else:
        print(f"\n  [Config {config}] Game-level reflection disabled — skipping")

    final_snap = prev_snapshot if prev_snapshot is not None else post_snap
    final_score = final_snap.score if final_snap else 0.0
    milestone_reached = result.startswith("SUCCESS")

    print(f"\n{'=' * 60}")
    print(f"  Game {game_id} ended: {result}")
    print(f"  Config: [{config}] {_CONFIG_LABELS[config]}")
    print(f"  Final score: {final_score:.1f}")
    if final_snap:
        ccr = final_snap.china_controlled / final_snap.china_total * 100
        print(f"  China control: {final_snap.china_controlled}/{final_snap.china_total} ({ccr:.0f}%)")
        print(f"  Japan China IPC: {final_snap.jp_china_ipc}")
        print(f"  China army remaining: {final_snap.china_army_value} IPC")
    print(f"  Rounds played: {current_round}")
    print(f"  Score trajectory: {[round(s, 1) for s in round_scores]}")
    print(f"  Plans reviewed: {len(reviews)}")
    for i, review in enumerate(reviews, 1):
        status = "ACHIEVED" if review.achieved else "FAILED"
        print(f"  {i}. [{status}] {review.plan_name}: {review.lesson[:80]}")
    print(f"{'=' * 60}\n")

    # ── Persist results to CSV for cross-game comparison ──
    _save_game_result(
        game_id=game_id,
        config=config,
        milestone_name=milestone["name"],
        milestone_reached=milestone_reached,
        result=result,
        rounds_played=current_round,
        final_score=final_score,
        final_snap=final_snap,
        round_scores=round_scores,
        reviews=reviews,
    )

    return result, reviews


def _save_game_result(
    game_id: str,
    config: str,
    milestone_name: str,
    milestone_reached: bool,
    result: str,
    rounds_played: int,
    final_score: float,
    final_snap,
    round_scores: list[float],
    reviews: list,
) -> None:
    """Append one row to results/game_results.csv for quantitative analysis."""
    import csv
    from datetime import datetime

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "game_results.csv")

    fieldnames = [
        "timestamp", "game_id", "config", "milestone", "milestone_reached",
        "result", "rounds_played", "final_score",
        "china_controlled", "china_total", "china_control_pct",
        "jp_china_ipc", "china_army_value",
        "score_r1", "score_r2", "score_r3", "score_r4", "score_r5", "score_r6",
        "plans_achieved", "plans_failed",
    ]

    file_exists = os.path.exists(csv_path)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "game_id": game_id,
        "config": config,
        "milestone": milestone_name,
        "milestone_reached": milestone_reached,
        "result": result,
        "rounds_played": rounds_played,
        "final_score": round(final_score, 1),
        "china_controlled": final_snap.china_controlled if final_snap else 0,
        "china_total": final_snap.china_total if final_snap else 0,
        "china_control_pct": round(
            final_snap.china_controlled / final_snap.china_total * 100, 1
        ) if final_snap and final_snap.china_total else 0,
        "jp_china_ipc": final_snap.jp_china_ipc if final_snap else 0,
        "china_army_value": final_snap.china_army_value if final_snap else 0,
        "plans_achieved": sum(1 for r in reviews if r.achieved),
        "plans_failed": sum(1 for r in reviews if not r.achieved),
    }

    for i in range(6):
        key = f"score_r{i + 1}"
        row[key] = round(round_scores[i], 1) if i < len(round_scores) else ""

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  [Results] Saved to {csv_path}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # ── Startup banner  ─────────────────────────────
    banner(4)
    effect(0)
    info(0,
         project="Reflexion Based Self-Learning Agent",
         version="2.1",
         environment="Development",
         description="",
         status="Initializing... "
         )

    # ── Argument parsing ───────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="TripleA LLM Agent — Hierarchical Reflexion",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="C",
        choices=["A", "B", "C", "a", "b", "c"],
        help=(
            "Ablation configuration:\n"
            "  A — Baseline: No reflection, no RAG (Round Plan only)\n"
            "  B — RAG:      Game-level reflection + experience retrieval\n"
            "  C — Full:     Reflexion + criticizer (default)"
        ),
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Clear experience pool before starting (use when switching configs)",
    )
    args = parser.parse_args()

    if args.clean:
        import shutil
        _exp_idx = os.path.join(PROJECT_ROOT, "knowledge", "exp_index")
        _exp_json = os.path.join(PROJECT_ROOT, "memory", "experiences.json")
        if os.path.exists(_exp_idx):
            shutil.rmtree(_exp_idx)
            print("[Clean] Deleted exp_index")
        if os.path.exists(_exp_json):
            os.remove(_exp_json)
            print("[Clean] Deleted experiences.json")

    run_game(milestone_id="m1", config=args.config.upper())
