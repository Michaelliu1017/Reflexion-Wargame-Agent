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
    load_national_strategy,
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


MILESTONES = [
    {
        "id": "m1",
        "name": "Capture India (Early Game Victory)",
        "check": lambda state: _owns(state, "India"),
        "max_rounds": 6,
    },
    {
        "id": "m2",
        "name": "Secure Southeast Asia Resource Zone",
        "check": lambda state: all(
            _owns(state, t)
            for t in ["Sumatra", "Java", "Borneo", "Celebes"]
        ),
        "max_rounds": 5,
    },
    {
        "id": "m3",
        "name": "Destroy US Navy",
        "check": lambda state: _us_fleet_destroyed(state),
        "max_rounds": 4,
    },
]


def _us_fleet_destroyed(state: dict) -> bool:
    """
    Check if the US fleet at Hawaiian Islands is eliminated.
    Considers the fleet destroyed when Hawaiian Islands has no US naval units.
    """
    for t in state.get("territories", []):
        if t.get("name") == "Hawaiian Islands":
            units = t.get("unitsSummary", {})
            naval = ["battleship", "carrier", "destroyer", "cruiser", "submarine", "transport"]
            us_naval = sum(units.get(u, 0) for u in naval)
            return us_naval == 0
    return False


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

def run_game(milestone_id: str = "m1"):
    """
    Run the game loop with Strategic Plan lifecycle:
    1. Load National Strategy → initialize plans
    2. Each round: RAG retrieve → Round Plan → execute → track progress
    3. Every 2 rounds: reassess plans
    4. Game end: reflexion per plan → update NS
    """
    milestone = next((m for m in MILESTONES if m["id"] == milestone_id), MILESTONES[0])
    print(f"\n{'=' * 60}")
    print(f"  Objective:  {milestone['name']}")
    print(f"  Max rounds: {milestone['max_rounds']}")
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

        # ── Initialize Strategic Plans (once, before round 1) ──
        if not plans_initialized:
            plan_tracker = _initialize_strategic_plans(ns, round_num=current_round)
            plans_initialized = True

        # ── Layer 3: Strategic Reassessment every 2 rounds ──
        if current_round >= 2 and current_round % 2 == 0:
            plan_tracker = _reassess_strategy(
                round_num=current_round,
                plan_tracker=plan_tracker,
                round_summaries=round_summaries,
            )

        # ── RAG retrieval: query by highest-priority active plan ──
        active_plans = [p for p in plan_tracker if p.status == "active"]
        active_plans.sort(key=lambda p: p.target_round)
        stage = "Early" if current_round <= 3 else ("Mid" if current_round <= 6 else "Late")
        if active_plans:
            top = active_plans[0]
            rag_query = f"Japan {stage} game: how to {top.name}"
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
        )
        game_log.extend(turn_log)

        # ── Round summary + plan progress tracking ──
        summary = _summarize_round(current_round, turn_log)
        round_summaries.append(summary)

        for p in plan_tracker:
            if p.status == "active":
                p.progress.append(f"Round {current_round}: {summary}")

        # ── Safety check: retry if turn didn't fully complete ──
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
                )
                game_log.extend(extra_log)
            except Exception:
                break

        # ── Milestone check ──
        state = _get_state_with_retry(max_wait_seconds=300)
        if state is None:
            print("  [Warning] Could not get post-turn state (Bridge timeout) — skipping milestone check")
            continue
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

    # ── Post-game Strategic Reflexion ──
    print(f"\n  Running Strategic Reflexion for game {game_id} (round {current_round})...")
    reviews = reflexion.reflect_and_store(
        game_id=game_id,
        game_log=game_log,
        result=result,
        round_num=current_round,
        plan_tracker=plan_tracker,
    )

    print(f"\n{'=' * 60}")
    print(f"  Game {game_id} ended: {result}")
    print(f"  Plans reviewed: {len(reviews)}")
    for i, review in enumerate(reviews, 1):
        status = "ACHIEVED" if review.achieved else "FAILED"
        print(f"  {i}. [{status}] {review.plan_name}: {review.lesson[:80]}")
    print(f"{'=' * 60}\n")

    return result, reviews


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
         version="1.0",
         environment="Development",
         description="",
         status="Initializing... "
         )

    # ── Argument parsing ───────────────────────────────────────
    parser = argparse.ArgumentParser(description="TripleA LLM Agent")
    parser.add_argument("--m1", action="store_true", help="Capture India (Early Game Victory)")
    parser.add_argument("--m2", action="store_true", help="Secure Southeast Asia Resource Zone")
    parser.add_argument("--m3", action="store_true", help="Destroy US Navy")
    args = parser.parse_args()

    if args.m2:
        milestone_id = "m2"
    elif args.m3:
        milestone_id = "m3"
    else:
        milestone_id = "m1"   # default

    run_game(milestone_id=milestone_id)
