"""
main.py

Main game loop.
- Each round runs the agent for Japan's full turn
- After each round, checks if milestone was reached
- On milestone reached (or timeout), triggers Reflexion and exits
"""
import os
import time
import uuid

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from agent import run_full_turn, LLM_MODEL
from bridge_client import TripleABridgeClient
from memory import GameMemory, ReflexionEngine

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
        "id": "capture_india",
        "name": "Capture Calcutta (Early Game Victory)",
        "check": lambda state: _owns(state, "India"),
        "max_rounds": 6,
    },
    {
        "id": "secure_sea_lanes",
        "name": "Secure Southeast Asia Resource Zone",
        "check": lambda state: all(
            _owns(state, t)
            for t in ["Sumatra", "Java", "Borneo", "Celebes"]
        ),
        "max_rounds": 5,
    },
    {
        "id": "destroy_us_navy",
        "name": "Destroy Pearl Harbor Fleet",
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

_MILESTONE_GOALS: dict[str, str] = {
    "capture_india":    "Southern Expansion",
    "secure_sea_lanes": "Southern Expansion",
    "destroy_us_navy":  "Pacific Dominance",
}


def run_game(milestone_id: str = "capture_india"):
    """
    Run the game loop for a single milestone objective.

    Args:
        milestone_id: which milestone to pursue (see MILESTONES above)
    """
    # Find the requested milestone
    milestone = next((m for m in MILESTONES if m["id"] == milestone_id), MILESTONES[0])
    strategic_goal = _MILESTONE_GOALS.get(milestone_id, "Southern Expansion")
    print(f"\n{'=' * 60}")
    print(f"  Objective:      {milestone['name']}")
    print(f"  Strategic Goal: {strategic_goal}")
    print(f"  Max rounds:     {milestone['max_rounds']}")
    print(f"{'=' * 60}\n")

    # Set up memory and reflexion
    memory = GameMemory(
        rules_path=os.path.join(PROJECT_ROOT, "knowledge", "rules.txt"),
        index_path=os.path.join(PROJECT_ROOT, "knowledge", "faiss_index"),
    )
    reflexion = ReflexionEngine(
        memory,
        experiences_path=os.path.join(PROJECT_ROOT, "memory", "experiences.json"),
        reflect_model=LLM_MODEL,
        critic_model="gpt-4o-mini",
    )

    game_id = str(uuid.uuid4())[:8]
    game_log = []
    current_round = 0

    while current_round < milestone["max_rounds"]:
        current_round += 1
        print(f"\n[Round {current_round} / {milestone['max_rounds']}]")

        # Wait for our turn (other players may be moving, or bridge may not be up yet)
        print("  Waiting for Japanese turn (Bridge must be running)...")
        dots = 0
        while not _client.is_our_turn():
            time.sleep(5)
            dots += 1
            if dots % 6 == 0:
                print("  Still waiting... ensure game + bridge are running on port 8081")

        # Detect which phase we're starting at (may be mid-turn if reconnecting)
        current_phase = _client.get_phase()
        print(f"  Japanese turn detected! Starting at phase: {current_phase}")

        stage = "Early" if current_round <= 3 else ("Mid" if current_round <= 6 else "Late")
        rag_context = memory.retrieve(
            f"[{stage}][{strategic_goal}] Japan round {current_round} strategy",
            k=2,
        )

        # Run full Japanese turn
        turn_log = run_full_turn(
            rag_context=rag_context,
            start_phase=current_phase,
            memory=memory,
            round_num=current_round,
            strategic_goal=strategic_goal,
        )
        game_log.extend(turn_log)

        # Safety check: if our turn didn't fully complete (rare race condition).
        # If other AIs ran fast and round N+1 has already started, skip — let main loop handle it.
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
                    memory=memory,
                    round_num=current_round,
                    strategic_goal=strategic_goal,
                )
                game_log.extend(extra_log)
            except Exception:
                break

        state = _get_state_with_retry(max_wait_seconds=300)
        if state is None:
            print("  [Warning] Could not get post-turn state (Bridge timeout) — skipping milestone check")
            continue
        if check_milestone(state, milestone):
            print(f"\n  Milestone REACHED: {milestone['name']}")
            result = f"SUCCESS - {milestone['name']}"
            break
    else:
        print(f"\n  Max rounds reached without achieving milestone.")
        result = f"TIMEOUT - {milestone['name']} not achieved in {milestone['max_rounds']} rounds"

    # Post-game Reflexion
    print(f"\n  Running Reflexion for game {game_id} (round {current_round})...")
    lessons = reflexion.reflect_and_store(
        game_id, game_log, result,
        round_num=current_round,
        strategic_goal=strategic_goal,
    )

    print(f"\n{'=' * 60}")
    print(f"  Game {game_id} ended: {result}")
    print(f"  Lessons learned: {len(lessons)}")
    for i, lesson in enumerate(lessons, 1):
        print(f"  {i}. {lesson}")
    print(f"{'=' * 60}\n")

    return result, lessons


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # ── Startup banner  ─────────────────────────────
    seed = "00080909000409490251409504612094067110920007719200086183000950930009923300095273000940940008454500080909"
    ORANGE, RESET = "\033[38;5;202m", "\033[0m"

    for i in range(13):
        s = int(seed[-8 * (i + 1):-8 * i or None])
        line = "".join(
            " " * (s // 10 ** j % 10) + f"{ORANGE}◆{RESET}" * (s // 10 ** (j + 1) % 10) for j in range(0, 8, 2))
        print(line)
        time.sleep(0.05)
    #*********风格4 — 竖条从中间展开
    import time

    width = 36
    for i in range(width // 2):
        left = width // 2 - i
        right = width // 2 + i
        line = " " * left + "\033[38;5;214m" + "█" * (right - left) + "\033[0m"
        print(f"\r{line}", end="", flush=True)
        time.sleep(0.03)
    print()
    #*************风格3 — 文字从噪点中浮现
    import time, random

    title = "  RRAgent  "
    chars = "**********"
    for step in range(15):
        line = ""
        for c in title.center(30):
            if random.random() < step / 14:
                line += f"\033[38;5;208m{c}\033[0m"
            else:
                line += f"\033[38;5;238m{random.choice(chars)}\033[0m"
        print("\r" + line, end="", flush=True)
        time.sleep(0.08)
    print()
    #*********************************8
    import random, time

    seed = "00080909000409490251409504612094067110920007719200086183000950930009923300095273000940940008454500080909"
    ORANGE, DIM, RESET = "\033[38;5;208m", "\033[38;5;202m", "\033[0m"
    noise = ["░", "▒", "▓", "·", ":", " "]

    # 先生成所有行的最终图案
    lines = []
    for i in range(13):
        s = int(seed[-8 * (i + 1):-8 * i or None])
        lines.append("".join(" " * (s // 10 ** j % 10) + "█" * (s // 10 ** (j + 1) % 10) for j in range(0, 8, 2)))

    width = max(len(l) for l in lines)

    # 噪点浮现：从全噪点逐渐收敛到真实图案
    for frame in range(20):
        output = []
        for line in lines:
            row = ""
            for ch in line.ljust(width):
                if ch == "█":
                    # 真实像素：随帧数增加，越来越稳定
                    if random.random() < frame / 19:
                        row += f"{ORANGE}█{RESET}"
                    else:
                        row += f"{DIM}{random.choice(noise)}{RESET}"
                else:
                    # 空白区域：随帧数增加，噪点越来越少
                    if random.random() < (1 - frame / 19) * 0.4:
                        row += f"{DIM}{random.choice(noise)}{RESET}"
                    else:
                        row += " "
            output.append(row)

        print("\n".join(output))
        time.sleep(0.06)
        print(f"\033[{len(lines)}A", end="")  # 光标回到顶部覆盖

    # 最终定格，干净输出
    for line in lines:
        s_clean = "".join(f"{ORANGE}█{RESET}" if c == "█" else " " for c in line.ljust(width))
        print(s_clean)



    # ── Argument parsing ───────────────────────────────────────
    parser = argparse.ArgumentParser(description="TripleA LLM Agent")
    parser.add_argument(
        "--milestone",
        choices=[m["id"] for m in MILESTONES],
        default="capture_india",
        help="Which milestone objective to pursue",
    )
    args = parser.parse_args()

    run_game(milestone_id=args.milestone)
