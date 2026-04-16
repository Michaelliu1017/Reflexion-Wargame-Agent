"""
agent.py
LangChain Agent: connects GPT with game tools.
GPT observes game state, makes decisions, and executes actions via tools.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

from bridge_client import TripleABridgeClient
from battle_predictor import predict_winrate
from display import (
    print_phase_header,
    print_rag_context,
    print_plan,
    print_transport_capacity,
)


# ─────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────

LLM_MODEL = "gpt-4o" #gpt-5.1

JAPAN_LOADING_ZONE = "6 Sea Zone"


# ─────────────────────────────────────────────────────────────
# Terminal colors
# ─────────────────────────────────────────────────────────────

class Colors:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    WHITE  = "\033[97m"
    BLUE   = "\033[94m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    MAGENTA = "\033[95m"


PHASE_COLORS = {
    "japanesePolitics":      Colors.WHITE,
    "japanesePurchase":      Colors.BLUE,
    "japaneseCombatMove":    Colors.RED,
    "japaneseBattle":        Colors.RED,
    "japaneseNonCombatMove": Colors.GREEN,
    "japanesePlace":         Colors.CYAN,
    "japaneseEndTurn":       Colors.CYAN,
}

# Maps internal phase names → display phase number and human-readable label
_PHASE_NUMBER: dict[str, int] = {
    "japanesePolitics":      1,
    "japanesePurchase":      2,
    "japaneseCombatMove":    3,
    "japaneseBattle":        3,  # sub-phase of combat
    "japaneseNonCombatMove": 4,
    "japanesePlace":         5,
    "japaneseEndTurn":       6,
}
_PHASE_DISPLAY_NAME: dict[str, str] = {
    "japanesePolitics":      "Politics",
    "japanesePurchase":      "Purchase Units",
    "japaneseCombatMove":    "Combat Move",
    "japaneseBattle":        "Battle",
    "japaneseNonCombatMove": "Noncombat Move",
    "japanesePlace":         "Place Units",
    "japaneseEndTurn":       "End Turn",
}


# ─────────────────────────────────────────────────────────────
# 自定义回调：彩色输出、抑制 tool_get_state JSON、详细战斗胜率
# ─────────────────────────────────────────────────────────────

class QuietCallbackHandler(BaseCallbackHandler):
    """
    Colorized, concise LangChain callback handler.
    - tool_get_state           → prints summary only (phase, PUs)
    - tool_predict_battle_odds → prints detailed win rate analysis with color
    - other tools              → truncated to 250 chars
    - on_llm_end               → prints LLM reasoning text (pre-tool-call thinking)
    - on_agent_action          → prints action log if non-empty
    Color is controlled by current_phase, updated per phase in run_full_turn.
    """

    def __init__(self):
        super().__init__()
        self._active: dict[str, str] = {}
        self.current_phase: str = ""
        self.move_tools_called: list[str] = []
        self.last_llm_output: str = ""   # stores last LLM reasoning for plan extraction

    def reset_phase_tracking(self) -> None:
        self.move_tools_called = []
        self.last_llm_output = ""

    # Color scheme: GREEN=Reflexion, CYAN=reasoning/thinking, WHITE=everything else
    _W = Colors.WHITE   # default output
    _G = Colors.GREEN   # Reflexion / RAG
    _C = Colors.CYAN    # agent reasoning / thinking

    def _c(self) -> str:
        # Always white — phase colors removed per user request
        return self._W

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        tool_name = serialized.get("name", "?")
        self._active[run_id] = tool_name
        if tool_name in ("tool_move_units", "tool_transport_units"):
            self.move_tools_called.append(tool_name)
        if tool_name == "tool_get_state":
            print(f"  {Colors.DIM}[→] tool_get_state(){Colors.RESET}")
        elif tool_name == "tool_predict_battle_odds":
            print(f"  {self._W}[→] tool_predict_battle_odds({input_str[:200]}){Colors.RESET}")
        else:
            short = input_str[:150] + "..." if len(input_str) > 150 else input_str
            print(f"  {self._W}[→] {tool_name}({short}){Colors.RESET}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        tool_name = self._active.pop(run_id, "?")

        if tool_name == "tool_get_state":
            try:
                first_line = str(output).split("\n")[0].strip("[]")
                print(f"  {Colors.DIM}[←] {first_line}{Colors.RESET}")
            except Exception:
                print(f"  {Colors.DIM}[←] game state OK{Colors.RESET}")

        elif tool_name == "tool_predict_battle_odds":
            try:
                data = json.loads(output) if isinstance(output, str) else output
                win_rate = data.get("win_rate", "?")
                advice   = data.get("advice", "")
                attacker = data.get("attacker", {})
                defender = data.get("defender", {})
                # Keep semantic colors only for win rate verdict
                if "ATTACK" in advice:
                    adv_c = Colors.GREEN
                elif "BORDERLINE" in advice:
                    adv_c = Colors.YELLOW
                else:
                    adv_c = Colors.RED
                print(f"  {'─'*48}")
                print(f"  ⚔  Battle Odds — Att:{attacker}  Def:{defender}")
                print(f"     Win Rate: {adv_c}{Colors.BOLD}{win_rate}{Colors.RESET}  →  {adv_c}{advice}{Colors.RESET}")
                print(f"  {'─'*48}")
            except Exception:
                print(f"  [←] tool_predict_battle_odds: {str(output)[:300]}")

        else:
            short = str(output)
            if len(short) > 250:
                short = short[:250] + "..."
            print(f"  {self._W}[←] {tool_name}: {short}{Colors.RESET}")

    def on_tool_error(self, error: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        tool_name = self._active.pop(run_id, "?")
        print(f"  {Colors.RED}[✗] {tool_name} error: {error}{Colors.RESET}")

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        pass

    def on_chat_model_start(
        self, serialized: dict, messages: list, **kwargs: Any
    ) -> None:
        total_chars = sum(
            len(str(getattr(msg, "content", "") or ""))
            for msg_list in messages
            for msg in msg_list
        )
        est_tokens = total_chars // 4
        print(f"  {Colors.DIM}[LLM] thinking... (~{est_tokens:,} tokens){Colors.RESET}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            if msg is None:
                return
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                text = content.strip()
                self.last_llm_output = text   # save for plan extraction
                display = text[:350] + "…" if len(text) > 350 else text
                print(f"  {self._C}[Reasoning] {display}{Colors.RESET}")
        except Exception:
            pass

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        output = finish.return_values.get("output", "")

        if self.current_phase == "japaneseCombatMove":
            # With return_direct=True, output is the tool return string.
            # The actual COMBAT PLAN is in last_llm_output (LLM's reasoning before calling tool_end_turn).
            plan_text = self.last_llm_output if self.last_llm_output.strip() else output
            print(f"\n  {self._C}{'═'*62}{Colors.RESET}")
            print(f"  {self._C}{Colors.BOLD}{'⚔  COMBAT PLAN + NCM PLAN':^62}{Colors.RESET}")
            print(f"  {self._C}{'═'*62}{Colors.RESET}")
            in_ncm = False
            for line in plan_text.strip().split("\n"):
                stripped = line.strip()
                if "NONCOMBAT PLAN" in stripped:
                    in_ncm = True
                    print(f"  {self._C}{Colors.BOLD}{line}{Colors.RESET}")
                elif "COMBAT PLAN" in stripped or "Strategic Direction" in stripped \
                        or "Primary Objective" in stripped:
                    print(f"  {self._C}{Colors.BOLD}{line}{Colors.RESET}")
                elif stripped.startswith("━"):
                    print(f"  {Colors.DIM}{line}{Colors.RESET}")
                elif in_ncm:
                    print(f"  {self._C}{line}{Colors.RESET}")
                else:
                    print(f"  {self._W}{line}{Colors.RESET}")
            print(f"  {self._C}{'═'*62}{Colors.RESET}\n")
        else:
            print(f"\n  {self._C}{Colors.BOLD}▶ Decision:{Colors.RESET} {self._W}{output[:500]}{Colors.RESET}")

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        # Track move tools here (on_tool_start may not fire in create_tool_calling_agent)
        tool_name = getattr(action, "tool", "")
        if tool_name in ("tool_move_units", "tool_transport_units"):
            self.move_tools_called.append(tool_name)
        log = getattr(action, "log", "").strip()
        if log and len(log) > 5:
            display = log[:300] + "…" if len(log) > 300 else log
            print(f"  {self._C}[Thinking] {display}{Colors.RESET}")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        pass

load_dotenv()

# 全局 bridge client 实例，所有工具共用这一个连接
_client = TripleABridgeClient()

# Purchase tracking for programmatic placement
_last_purchase: list[dict] = []

# Transport usage tracking per phase — counts how many transports were dispatched this NCM
_transports_used_this_phase: int = 0


# ─────────────────────────────────────────────────────────────
# 游戏状态压缩工具（减少 token 消耗）
# ─────────────────────────────────────────────────────────────

_UNIT_SHORT: dict[str, str] = {
    "infantry":        "inf",
    "artillery":       "art",
    "armour":          "arm",
    "mech_infantry":   "mec",
    "fighter":         "ftr",
    "tactical_bomber": "tacbmb",
    "bomber":          "strbmb",
    "transport":       "trn",
    "destroyer":       "des",
    "submarine":       "sub",
    "carrier":         "car",
    "battleship":      "bbs",
    "cruiser":         "cru",
    "aaGun":           "aag",
}

# Canonical unit type names accepted by the Java Bridge.
# Maps any abbreviation / alternate name the LLM might use → correct name.
_UNIT_CANONICAL: dict[str, str] = {
    # compressed state abbreviations
    "inf":          "infantry",
    "art":          "artillery",
    "arm":          "armour",
    "mec":          "mech_infantry",
    "ftr":          "fighter",
    "tac":          "tactical_bomber",
    "tacbmb":       "tactical_bomber",
    "bmb":          "bomber",
    "strbmb":       "bomber",
    "trn":          "transport",
    "des":          "destroyer",
    "sub":          "submarine",
    "car":          "carrier",
    "bbs":          "battleship",
    "cru":          "cruiser",
    "aag":          "aaGun",
    # common LLM hallucinations
    "tac_bomber":       "tactical_bomber",
    "tactical_bombers": "tactical_bomber",
    "armor":            "armour",
    "tank":             "armour",
    "tanks":            "armour",
    "mechanized":       "mech_infantry",
    "mech":             "mech_infantry",
    "aa_gun":           "aaGun",
    "aa":               "aaGun",
    "anti_air":         "aaGun",
    "strat_bomber":     "bomber",
    "strategic_bomber": "bomber",
    "strat":            "bomber",
    "fighters":         "fighter",
    "infantries":       "infantry",
}


def _normalize_units(units: list) -> list:
    """Normalize unitType in every unit entry using _UNIT_CANONICAL aliases."""
    result = []
    for u in units:
        item = dict(u)
        raw = item.get("unitType", "")
        item["unitType"] = _UNIT_CANONICAL.get(raw, raw)
        result.append(item)
    return result
_INFRA_KEYS = {"harbour", "airfield", "factory_major", "factory_minor"}


def _fmt_units(units: dict) -> str:
    """Convert unit dict to compact string, e.g. 'inf×3 art×1'."""
    parts = []
    for k, v in units.items():
        if v > 0 and k not in _INFRA_KEYS:
            parts.append(f"{_UNIT_SHORT.get(k, k)}×{v}")
    return " ".join(parts) if parts else "(empty)"


class _StateCache:
    """
    Caches the last game state snapshot (territory → units mapping).
    On second+ call, returns only the diff instead of the full state.
    Reset at the start of each phase to guarantee a fresh full state.

    Committed tracking: records units that have already moved this phase.
    These units are annotated in state output so the LLM knows they cannot
    move again (critical for Combat Move phase).
    """

    def __init__(self):
        self._prev_jp: dict[str, dict] = {}
        self._prev_enemy: dict[str, dict] = {}
        self._prev_pus: int | str = "?"
        self._has_prev = False
        self._committed: dict[str, dict[str, int]] = {}

    def reset(self) -> None:
        self._prev_jp = {}
        self._prev_enemy = {}
        self._prev_pus = "?"
        self._has_prev = False
        self._committed = {}

    def record_committed(self, to_territory: str, units: list[dict]) -> None:
        """Record units that arrived at to_territory via a move this phase."""
        if to_territory not in self._committed:
            self._committed[to_territory] = {}
        for u in units:
            ut = u.get("unitType", "")
            cnt = u.get("count", 1)
            self._committed[to_territory][ut] = self._committed[to_territory].get(ut, 0) + cnt

    def _committed_annotation(self, territory: str) -> str:
        """Return annotation string for committed units in a territory, or empty."""
        c = self._committed.get(territory)
        if not c:
            return ""
        parts = [f"{_UNIT_SHORT.get(k, k)}×{v}" for k, v in c.items() if v > 0]
        if not parts:
            return ""
        return f" ⚠COMMITTED({' '.join(parts)} already moved, CANNOT move again)"

    def get_state_text(self, state: dict) -> str:
        if not self._has_prev:
            self._snapshot(state)
            self._has_prev = True
            base = _compress_state_for_llm(state)
            return self._inject_committed_annotations(base)

        new_jp, new_enemy, new_pus = self._extract(state)
        diff_lines = self._diff(new_jp, new_enemy, new_pus)
        self._prev_jp = new_jp
        self._prev_enemy = new_enemy
        self._prev_pus = new_pus

        if not diff_lines:
            return "[State Update] No changes since last check."

        header = f"[State Update — {len(diff_lines)} change(s) since last check]"
        return header + "\n" + "\n".join(diff_lines)

    def _inject_committed_annotations(self, state_text: str) -> str:
        """Append COMMITTED annotations to territory lines in the full state text."""
        if not self._committed:
            return state_text
        lines = state_text.split("\n")
        out = []
        for line in lines:
            annotated = False
            for terr in self._committed:
                if line.strip().startswith(terr + ":") or f"] {terr}:" in line or f" {terr}:" in line:
                    ann = self._committed_annotation(terr)
                    if ann:
                        out.append(line + ann)
                        annotated = True
                        break
            if not annotated:
                out.append(line)
        return "\n".join(out)

    def _snapshot(self, state: dict) -> None:
        self._prev_jp, self._prev_enemy, self._prev_pus = self._extract(state)

    @staticmethod
    def _extract(state: dict) -> tuple[dict[str, dict], dict[str, dict], int | str]:
        units_by_terr = state.get("unitsByTerritory", {})
        jp_pus = state.get("japan", {}).get("pus", "?")

        jp_map: dict[str, dict] = {}
        enemy_map: dict[str, dict] = {}

        for t in state.get("territories", []):
            name = t.get("name", "")
            owner = t.get("owner", "") or ""
            is_water = t.get("isWater", False)
            is_jp = owner in ("Japanese", "Japan")

            jp_units = {k: v for k, v in units_by_terr.get(name, {}).items()
                        if v > 0 and k not in _INFRA_KEYS}
            all_units = {k: v for k, v in t.get("unitsSummary", {}).items()
                         if v > 0 and k not in _INFRA_KEYS}

            if is_jp and (jp_units or not is_water):
                jp_map[name] = jp_units
            elif owner and owner not in ("Neutral", "") and all_units and not is_water:
                enemy_map[name] = all_units

        return jp_map, enemy_map, jp_pus

    def _diff(
        self,
        new_jp: dict[str, dict],
        new_enemy: dict[str, dict],
        new_pus: int | str,
    ) -> list[str]:
        lines: list[str] = []

        if str(new_pus) != str(self._prev_pus):
            lines.append(f"  △ Japan PUs: {self._prev_pus} → {new_pus}")

        lines.extend(self._diff_territory_group(self._prev_jp, new_jp, "JP"))
        lines.extend(self._diff_territory_group(self._prev_enemy, new_enemy, "Enemy"))
        return lines

    def _diff_territory_group(
        self,
        old: dict[str, dict],
        new: dict[str, dict],
        label: str,
    ) -> list[str]:
        lines: list[str] = []
        all_names = sorted(set(old) | set(new))
        for name in all_names:
            old_units = old.get(name, {})
            new_units = new.get(name, {})
            if old_units == new_units:
                continue
            old_str = _fmt_units(old_units) if old_units else "(empty)"
            new_str = _fmt_units(new_units) if new_units else "(empty)"
            ann = self._committed_annotation(name)
            lines.append(f"  △ [{label}] {name}: {old_str} → {new_str}{ann}")
        return lines


_state_cache = _StateCache()


def _compress_state_for_llm(state: dict) -> str:
    """
    Compress full game state JSON to a structured text summary.
    Raw JSON ~3000-6000 tokens → compressed ~400-900 tokens.

    Bridge field names (from BridgeStateBuilder.java):
      game.stepName        — current phase
      game.round           — round number
      japan.pus            — Japan PUs
      territories[].unitsSummary — all units in territory (all owners)
      territories[].neighbors    — adjacent territory list
      territories[].isWater      — whether sea zone
      territories[].puValue      — IPC value
      unitsByTerritory     — Japanese units only, keyed by territory name
      purchaseOptions      — buyable units
      placeOptions         — placement locations (territory + maxPlaceCapacity)
    """
    game      = state.get("game", {})
    phase     = game.get("stepName", "?")
    round_num = game.get("round", "?")
    jp_pus    = state.get("japan", {}).get("pus", "?")

    lines = [f"[Round:{round_num} | Phase:{phase} | Japan PUs:{jp_pus}]", ""]

    units_by_terr: dict[str, dict] = state.get("unitsByTerritory", {})

    jp_land:    list[tuple[str, dict, list, int]] = []
    jp_sea:     list[tuple[str, dict, list]]      = []
    enemy_land: list[tuple[str, str, dict, int, list]] = []

    for t in state.get("territories", []):
        name      = t.get("name", "")
        owner     = t.get("owner", "") or ""
        is_water  = t.get("isWater", False)
        neighbors = t.get("neighbors", [])
        pu_value  = t.get("puValue", 0)

        jp_units = {k: v for k, v in units_by_terr.get(name, {}).items()
                    if v > 0 and k not in _INFRA_KEYS}

        all_units = {k: v for k, v in t.get("unitsSummary", {}).items()
                     if v > 0 and k not in _INFRA_KEYS}

        is_jp = owner in ("Japanese", "Japan")

        if is_jp:
            if is_water:
                if jp_units:
                    jp_sea.append((name, jp_units, neighbors))
            else:
                jp_land.append((name, jp_units, neighbors, pu_value))
        elif owner and owner not in ("Neutral", "") and all_units and not is_water:
            enemy_land.append((name, owner, all_units, pu_value, neighbors))

    if jp_land:
        lines.append("■ Japanese Land Territories")
        for name, units, neighbors, pu in jp_land:
            u_str = _fmt_units(units) if units else "(no units)"
            pu_str = f"[{pu}IPC] " if pu else ""
            n_str = ", ".join(neighbors[:8])
            lines.append(f"  {pu_str}{name}: {u_str}")
            lines.append(f"    neighbors→ {n_str}")
        lines.append("")

    if jp_sea:
        lines.append("■ Japanese Sea Zones (with Japanese ships)")
        for name, units, neighbors in jp_sea:
            n_str = ", ".join(neighbors[:6])
            lines.append(f"  {name}: {_fmt_units(units)}")
            lines.append(f"    neighbors→ {n_str}")
        lines.append("")

    if enemy_land:
        lines.append("■ Enemy Land Territories")
        for name, owner, units, pu, neighbors in enemy_land:
            pu_str = f"[{pu}IPC] " if pu else ""
            n_str = ", ".join(neighbors[:6])
            lines.append(f"  {pu_str}{name} [{owner}]: {_fmt_units(units)}")
            lines.append(f"    neighbors→ {n_str}")
        lines.append("")

    if "Purchase" in phase or "purchase" in phase:
        purchase_opts = state.get("purchaseOptions", [])
        if purchase_opts:
            lines.append("■ Available Purchases (purchaseOptions)")
            for opt in purchase_opts[:18]:
                ut       = opt.get("unitType", "?")
                cost     = opt.get("cost", "?")
                max_aff  = opt.get("maxAffordable", "?")
                lines.append(f"  {ut}: {cost} PUs (max affordable: {max_aff})")
            lines.append("")

    if "Place" in phase or "place" in phase:
        place_opts = state.get("placeOptions", [])
        if place_opts:
            lines.append("■ Placement Options (placeOptions) — only new units purchased this turn go here")
            for opt in place_opts:
                terr     = opt.get("territory", "?")
                max_cap  = opt.get("maxPlaceCapacity")
                cap_str  = f" (capacity limit: {max_cap})" if max_cap else ""
                lines.append(f"  → {terr}{cap_str}")
            lines.append("")

    lines.append(
        "⚠ UNIT TYPE NAMES IN TOOLS: Always use the FULL canonical name in tool calls: "
        "infantry | artillery | armour | mech_infantry | fighter | tactical_bomber | bomber | "
        "transport | destroyer | submarine | carrier | battleship | cruiser | aaGun"
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Transport capacity helper (Bug 3 fix)
# ─────────────────────────────────────────────────────────────

# Unit types that live at sea / in the air — not land cargo on a transport
_SEA_AIR_UNIT_TYPES = frozenset({
    "transport", "destroyer", "submarine", "carrier",
    "battleship", "cruiser", "fighter", "tactical_bomber",
    "strategic_bomber", "bomber", "aaGun",
})


def _get_ground_units_in_sz(sz_name: str) -> list[dict]:
    """Query current state for all Japanese ground units in a sea zone (cargo on transports)."""
    try:
        state = _client.get_state()
        units = state.get("unitsByTerritory", {}).get(sz_name, {})
        result = []
        for ut, count in units.items():
            if ut not in _SEA_AIR_UNIT_TYPES and count > 0:
                result.append({"unitType": ut, "count": count})
        return result
    except Exception:
        return []


def compute_transport_capacity(state: dict) -> str:
    """
    Scan all Japanese sea zones for transports.
    Returns a summary string ready to inject into the LLM prompt.
    Pure function with no side effects.
    """
    units_by_terr: dict[str, dict] = state.get("unitsByTerritory", {})
    terr_map: dict[str, dict] = {
        t["name"]: t for t in state.get("territories", [])
    }

    lines: list[str] = []
    for terr_name, units in units_by_terr.items():
        t_info = terr_map.get(terr_name, {})
        if not t_info.get("isWater", False):
            continue
        transport_count = units.get("transport", 0)
        if transport_count <= 0:
            continue

        # Detect whether transports are carrying land units
        ground = {
            k: v for k, v in units.items()
            if k not in _SEA_AIR_UNIT_TYPES and v > 0
        }
        loaded_str = (
            "LOADED: " + ", ".join(f"{k}×{v}" for k, v in ground.items())
            if ground else "EMPTY"
        )

        # Adjacent land territories (for context)
        adj_land = [
            n for n in t_info.get("neighbors", [])
            if not terr_map.get(n, {}).get("isWater", True)
        ][:5]
        adj_str = f" | adj land: {', '.join(adj_land)}" if adj_land else ""

        lines.append(
            f"  {terr_name}: {transport_count} transport(s) [{loaded_str}]"
            f" — capacity {transport_count} heavy + {transport_count} light units{adj_str}"
        )

    return "\n".join(lines) if lines else "  No Japanese transports found in any sea zone."


# ─────────────────────────────────────────────────────────────
# Combat plan extractor — parses agent output after Phase 3
# ─────────────────────────────────────────────────────────────

def _extract_combat_plan(output: str) -> str:
    """
    Extract the COMBAT PLAN block from the agent's output.
    Stops before the NONCOMBAT PLAN section so they remain separate.
    """
    if not output:
        return ""
    for marker in ("COMBAT PLAN", "Combat Plan —", "PLAN:"):
        idx = output.find(marker)
        if idx >= 0:
            text = output[idx:]
            # Stop before NONCOMBAT PLAN section
            ncm_idx = text.find("NONCOMBAT PLAN")
            if ncm_idx > 0:
                text = text[:ncm_idx]
            return text.strip()
    return ""


def _extract_noncombat_plan(output: str) -> str:
    """Extract the NONCOMBAT PLAN block from the agent's final output text."""
    if not output:
        return ""
    idx = output.find("NONCOMBAT PLAN")
    if idx >= 0:
        return output[idx:].strip()
    return ""


# ─────────────────────────────────────────────────────────────
# 工具定义
# @tool 让 GPT 能"看到"并调用这些函数
# docstring 是 GPT 判断要不要调这个工具的依据，要写清楚
# ─────────────────────────────────────────────────────────────

@tool
def tool_get_state() -> str:
    """
    Get current game state summary.
    Includes: round/phase, Japan PUs, all territory owners and units, purchasable/placeable units.
    Always call this first before making any decision.
    First call each phase returns full state; subsequent calls return only changes.
    """
    state = _client.get_state()
    return _state_cache.get_state_text(state)


@tool
def tool_get_legal_actions() -> str:
    """
    Get all legal actions available in the current phase.
    Returns action types such as BUY_UNITS, PLACE_UNITS, PERFORM_MOVE, END_TURN.
    Call before executing any action to confirm what is available.
    """
    actions = _client.get_legal_actions()
    return json.dumps(actions, ensure_ascii=False, indent=2)


@tool
def tool_predict_battle_odds(attacker_json: str, defender_json: str) -> str:
    """
    Call before attacking: Monte Carlo simulator (1000 rounds) predicts attacker win probability.
    Accuracy ±1.5%. Results are fully reliable — trust them for all combat decisions.
    attacker_json: attacker units, format {"infantry": 3, "armour": 1}
    defender_json: defender units, format {"infantry": 2, "aaGun": 1}
    Supported types: infantry, mech_infantry, artillery, armour, fighter,
                     tactical_bomber, strategic_bomber, aa_gun
    Decision thresholds:
      win rate ≥ 55% → attack (clear advantage)
      win rate 40-55% → borderline: weigh strategic value
      win rate < 40% → strongly advise against; consolidate in NCM
    """
    try:
        attacker = json.loads(attacker_json) if isinstance(attacker_json, str) else attacker_json
        defender = json.loads(defender_json) if isinstance(defender_json, str) else defender_json
        prob = predict_winrate(attacker, defender)
        pct = round(prob * 100, 1)
        if prob >= 0.55:
            advice = "ATTACK — clear advantage"
        elif prob >= 0.40:
            advice = "BORDERLINE — weigh strategic value carefully"
        else:
            advice = "HOLD — consolidate more forces in NCM first"
        return json.dumps({
            "win_rate": f"{pct}%",
            "advice": advice,
            "simulator": "Monte Carlo 1000 rounds, ±1.5% error, fully reliable",
            "attacker": attacker,
            "defender": defender,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(return_direct=True)
def tool_end_turn() -> str:
    """
    *** FINAL action in every phase. Call this LAST. ***

    Ends the current phase and advances the game to the next one.
    This tool returns your phase summary directly as the Final Answer.
    After calling this tool, execution stops automatically — do NOT call any other tool.
    """
    result = _client.act_end_turn()
    ok = result.get("ok", False)
    status = "Phase ended successfully." if ok else f"End turn failed: {result.get('error', 'unknown')}"
    return status


@tool
def tool_buy_units(items_json: Any) -> str:
    """
    Purchase phase: buy units.
    items_json: JSON string OR Python list. Format: [{"unitType": "infantry", "count": 3}, {"unitType": "armour", "count": 1}]
    Available unit types (cost in IPC): infantry(3), artillery(4), armour(6), fighter(10),
      tactical_bomber(11), destroyer(8), submarine(6), transport(7), carrier(16), battleship(20)
    Total cost must not exceed current PUs.
    """
    if isinstance(items_json, (list, dict)):
        items = items_json if isinstance(items_json, list) else [items_json]
    else:
        try:
            items = json.loads(items_json)
        except (json.JSONDecodeError, TypeError) as e:
            return json.dumps({"ok": False, "error": f"JSON parse error: {e}"})
    result = _client.act({"type": "BUY_UNITS", "items": items})
    global _last_purchase
    if result.get("ok"):
        _last_purchase = list(items)
    return json.dumps(result, ensure_ascii=False)


@tool
def tool_place_units(placements_json: Any) -> str:
    """
    Place phase: deploy units purchased this turn onto the map.
    placements_json: JSON string OR Python list. Format: [{"territory": "Japan", "unitType": "infantry", "count": 3}]
    Can only place in territories listed in placeOptions (must have a factory).
    Count must match exactly what was purchased this turn.
    """
    if isinstance(placements_json, (list, dict)):
        placements = placements_json if isinstance(placements_json, list) else [placements_json]
    else:
        try:
            placements = json.loads(placements_json)
        except (json.JSONDecodeError, TypeError) as e:
            return json.dumps({"ok": False, "error": f"JSON parse error: {e}"})
    result = _client.act({"type": "PLACE_UNITS", "placements": placements})
    return json.dumps(result, ensure_ascii=False)


@tool
def tool_transport_units(
    load_from: str,
    load_sz: str,
    transit_sz: str,
    units_json: str,
) -> str:
    """
    Noncombat Move — Step 1: load land units onto a transport and move it to a transit sea zone.

    This is a 2-round operation:
      - This NCM: call this tool → load units + transport moves to transit_sz
      - Next NCM: call tool_move_units(transit_sz, land_territory, units_json) to land troops
        Then immediately call tool_move_units(transit_sz, "6 Sea Zone", transport) to return

    Transport capacity: 1 heavy + 1 light (max 1 heavy per transport)
      Heavy units: armour, artillery, mech_infantry
      Light units: infantry
      Best combo: 1 armour + 1 infantry (max combat value per transport)

    load_from:  coastal territory where land units are (e.g. "Japan")
    load_sz:    sea zone with Japanese transport, adjacent to load_from (e.g. "6 Sea Zone")
    transit_sz: destination sea zone, adjacent to load_sz AND adjacent to target land (e.g. "19 Sea Zone")
    units_json: units to load, format: [{"unitType": "infantry", "count": 1}]

    Before calling, verify with tool_get_state:
    1. load_from has sufficient target units
    2. load_sz has a Japanese transport
    3. load_sz is adjacent to load_from; transit_sz is adjacent to load_sz
    4. Sea zone name format: number first, e.g. "6 Sea Zone", "19 Sea Zone"
    """
    try:
        units = json.loads(units_json) if isinstance(units_json, str) else units_json
    except json.JSONDecodeError as e:
        return json.dumps({"ok": False, "error": f"JSON parse error: {e}"})

    units = _normalize_units(units)

    # Pre-check: is there an available (unused) transport at load_sz?
    global _transports_used_this_phase
    try:
        _ubt_pre = _client.get_state().get("unitsByTerritory", {})
        _sz_units = _ubt_pre.get(load_sz, {})
        transports_here = _sz_units.get("transport", 0)
        available = transports_here - _transports_used_this_phase
        if transports_here == 0:
            return json.dumps({
                "ok": False,
                "error": f"No transports at {load_sz}. Cannot load. STOP loading and call tool_end_turn().",
            }, ensure_ascii=False)
        if available <= 0:
            return json.dumps({
                "ok": False,
                "error": (
                    f"All {transports_here} transport(s) at {load_sz} already used this round "
                    f"(movement exhausted). STOP loading and call tool_end_turn()."
                ),
            }, ensure_ascii=False)
    except Exception:
        pass

    # Tank-first enforcement: if origin has tanks but load doesn't include one, reject
    has_tank_in_load = any(u.get("unitType") == "armour" for u in units)
    if not has_tank_in_load:
        try:
            _ubt = _client.get_state().get("unitsByTerritory", {})
            _origin_units = _ubt.get(load_from, {})
            tanks_available = _origin_units.get("armour", 0)
            if tanks_available > 0:
                return json.dumps({
                    "ok": False,
                    "error": (
                        f"Blocked: {load_from} has {tanks_available} tank(s) available but you're loading infantry only. "
                        f"Tanks have 3x the attack power of infantry — ALWAYS load 1 armour + 1 infantry per transport. "
                        f'Fix: [{{"unitType":"armour","count":1}},{{"unitType":"infantry","count":1}}]'
                    ),
                }, ensure_ascii=False)
        except Exception:
            pass

    # Step A: load — move land units from coastal territory to adjacent sea zone
    r_load = _client.act_move(load_from, load_sz, units)
    if not r_load.get("ok", False):
        return json.dumps({
            "ok": False,
            "error": f"Load failed: {r_load.get('error', 'unknown error')}",
            "step": "load",
            "hint": f"Verify {load_sz} is adjacent to {load_from} and has a Japanese transport. Max 1 heavy + 1 light per transport.",
        }, ensure_ascii=False)

    # Step B: transport move sea → sea
    # Must include ALL ground units in the sea zone (not just newly loaded ones),
    # otherwise the game engine rejects with "Transports cannot leave their units".
    all_ground_in_sz = _get_ground_units_in_sz(load_sz)
    if all_ground_in_sz:
        transport_and_units = all_ground_in_sz + [{"unitType": "transport", "count": 1}]
    else:
        transport_and_units = units + [{"unitType": "transport", "count": 1}]

    r_move = _client.act_move(load_sz, transit_sz, transport_and_units)
    if not r_move.get("ok", False):
        game_err = r_move.get("error", "unknown error")
        print(
            f"  {Colors.RED}[Transport] Step B failed: {load_sz}→{transit_sz} "
            f"transport+{transport_and_units}  reason: {game_err}{Colors.RESET}"
        )
        return json.dumps({
            "ok": False,
            "error": (
                f"Transport move failed (Step B): {game_err}. "
                f"Units are now in {load_sz} but transport did not move. "
                f"Do not retry — skip this transport operation and call tool_end_turn()."
            ),
            "step": "transit",
            "loaded_ok": True,
            "loaded_units": units,
            "loaded_at": load_sz,
        }, ensure_ascii=False)

    _transports_used_this_phase += 1

    # Calculate remaining idle transports at load_sz
    _remaining = 0
    try:
        _ubt_post = _client.get_state().get("unitsByTerritory", {})
        _remaining = _ubt_post.get(load_sz, {}).get("transport", 0) - _transports_used_this_phase
        _remaining = max(_remaining, 0)
    except Exception:
        pass

    print(
        f"  {Colors.GREEN}[Transport] Loaded: {load_from} → {load_sz} → {transit_sz}"
        f"  cargo: {units}{Colors.RESET}"
    )
    return json.dumps({
        "ok": True,
        "message": (
            f"Load+move success: {units} from {load_from} are aboard transport at {transit_sz}. "
            f"Remaining idle transports at {load_sz}: {_remaining}. "
            + (f"You can dispatch {_remaining} more transport(s) this round."
               if _remaining > 0
               else "All transports dispatched — do NOT call tool_transport_units again this round.")
        ),
        "transit_sz": transit_sz,
        "units": units,
        "remaining_transports": _remaining,
    }, ensure_ascii=False)


@tool
def tool_move_units(from_territory: str, to_territory: str, units_json: str) -> str:
    """
    Move units from one territory (or sea zone) to another.

    Use case 1 — standard land/air move (Combat Move or Noncombat Move):
      Moving to an enemy territory during Combat Move triggers combat.
      During Noncombat Move, only move to friendly or neutral territories.

    Use case 2 — amphibious landing (Noncombat Move, second step of transport cycle):
      Last round you loaded troops to a transit sea zone (e.g. "19 Sea Zone") via tool_transport_units.
      This round in NCM, land them on adjacent territory:
        tool_move_units("19 Sea Zone", "Kiangsu", '[{"unitType":"infantry","count":1}]')
      List only land units, NOT the transport.
      Target must be adjacent to the sea zone and be friendly or neutral.

    Use case 3 — empty transport return (after unloading):
      tool_move_units("<transit_sz>", "6 Sea Zone", '[{"unitType":"transport","count":1}]')
      This is the ONLY case where moving an empty transport is allowed.

    from_territory: origin (territory or sea zone name — must match get_state exactly)
    to_territory:   destination (territory or sea zone name — must match get_state exactly)
    units_json:     format: [{"unitType": "infantry", "count": 2}]
    Call multiple times for different unit groups. Call tool_end_turn when done.
    """
    try:
        units = json.loads(units_json) if isinstance(units_json, str) else units_json
    except json.JSONDecodeError as e:
        return json.dumps({"ok": False, "error": f"JSON parse error: {e}"})

    units = _normalize_units(units)
    unit_types = [u.get("unitType", "") for u in units]
    only_transports = unit_types and all(t == "transport" for t in unit_types)
    if only_transports:
        if to_territory == JAPAN_LOADING_ZONE:
            print(
                f"  {Colors.DIM}[Transport] Empty transport returning to {JAPAN_LOADING_ZONE}...{Colors.RESET}"
            )
        else:
            return json.dumps({
                "ok": False,
                "error": (
                    f"Blocked: empty transport cannot move to {to_territory}. "
                    f"Empty transports may only return to the loading zone ({JAPAN_LOADING_ZONE}). "
                    f"To project forces, first use tool_transport_units to load troops, then the loaded transport moves. "
                    f"Example: tool_transport_units('Japan', '{JAPAN_LOADING_ZONE}', '{to_territory}', units_json)"
                ),
            }, ensure_ascii=False)

    # Owner validation during Combat Move: prevent attacking own territories
    current_phase = ""
    try:
        current_phase = _client.get_phase()
    except Exception:
        pass
    if current_phase == "japaneseCombatMove" and "Sea Zone" not in to_territory:
        try:
            state = _state_cache.get()
            if state is None:
                state = _client.get_state()
            territories = state.get("territories", [])
            owner = ""
            for _t in territories:
                if _t.get("name") == to_territory:
                    owner = _t.get("owner", "")
                    break
            if owner.lower() in ("japanese", "japan"):
                return json.dumps({
                    "ok": False,
                    "error": (
                        f"Blocked: {to_territory} is owned by {owner}. "
                        f"Combat Move cannot target friendly territories. "
                        f"Use Noncombat Move to reposition within friendly territory."
                    ),
                }, ensure_ascii=False)
        except Exception:
            pass

    # Auto-complete unload: when moving from sea zone to land, unload ALL ground units
    if "Sea Zone" in from_territory and "Sea Zone" not in to_territory and not only_transports:
        all_ground = _get_ground_units_in_sz(from_territory)
        if all_ground:
            agent_types = {u.get("unitType") for u in units}
            full_types = {u.get("unitType") for u in all_ground}
            if full_types - agent_types:
                missing = [u for u in all_ground if u.get("unitType") not in agent_types]
                missing_str = ", ".join(f"{u['count']}x {u['unitType']}" for u in missing)
                print(
                    f"  {Colors.YELLOW}[Unload] Agent forgot {missing_str} — auto-including all ground units{Colors.RESET}"
                )
            units = all_ground

    result = _client.act_move(from_territory, to_territory, units)

    if result.get("ok", False):
        _state_cache.record_committed(to_territory, units)
    else:
        err = result.get("error", "")
        result["hint"] = (
            f"Move {from_territory} → {to_territory} FAILED. "
            f"DO NOT retry this exact move — the units are likely already committed or the territories are not adjacent. "
            f"Skip and move to your next action."
        )

    return json.dumps(result, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# System Prompt
# {rag_context} 是占位符，Step 3 完成后会注入规则和经验
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional player controlling Japan in Axis & Allies Pacific 1940.
You interact with the game ONLY through tools. Never guess the board state — always call tool_get_state first.

╔══════════════════════════════════════════════════════════════╗
║  CRITICAL PROTOCOL — READ BEFORE ANYTHING ELSE               ║
╠══════════════════════════════════════════════════════════════╣
║  tool_end_turn() is the ABSOLUTE FINAL action in every       ║
║  phase. The moment it returns {{"ok": true}}:                ║
║    1. Write your phase summary (plain text).                 ║
║    2. STOP. Do NOT call tool_get_state or any other tool.   ║
║    3. The orchestrator handles the next phase for you.       ║
║  Calling ANY tool after tool_end_turn = protocol violation.  ║
╚══════════════════════════════════════════════════════════════╝

=== JAPAN TURN PHASES (in order) ===
1. Politics      — declare war or pass
2. Purchase      — buy units (BUY_UNITS → END_TURN → STOP)
3. Combat Move   — attack positions (PERFORM_MOVE × N → END_TURN → STOP)
4. Battle        — resolved automatically
5. Noncombat Move— reposition/reinforce (PERFORM_MOVE × N → END_TURN → STOP)
6. Place Units   — handled automatically (no LLM needed)
7. Collect Income— automatic

You handle ONE phase per invocation. After END_TURN succeeds, your job is done.

=== UNIT STATS ===
infantry (inf)      3 PU  | Att 1  Def 2  Mov 1 | light unit (1 slot)
artillery (art)     4 PU  | Att 2  Def 2  Mov 1 | heavy unit (1 slot) | boosts 1 adjacent inf to Att 2
armour/tank         6 PU  | Att 3  Def 3  Mov 2 | heavy unit (1 slot)
fighter             10 PU | Att 3  Def 4  Mov 4 | air unit — can land on carrier
tac_bomber          11 PU | Att 3  Def 3  Mov 4 | +1 Att when paired with fighter or tank
strategic_bomber    12 PU | Att 4  Def 1  Mov 6 | can strategic-bomb enemy factories
destroyer           8 PU  | Att 2  Def 2  Mov 2 | counters submarines
transport           7 PU  | Att —  Def —  Mov 2 | carries 1 heavy + 1 light (or 2 light)
carrier             16 PU | Att 1  Def 2  Mov 2 | holds 2 fighters
battleship          20 PU | Att 4  Def 4  Mov 2 | takes 2 hits

Transport capacity: 1 heavy + 1 light (max 1 heavy unit per transport):
  BEST: 1 armour + 1 inf (9 PU, Att 4) | 1 art + 1 inf (7 PU, Att 3) | 2 infantry (6 PU, Att 2)

=== AIRCRAFT USAGE — CRITICAL ===
Fighters and tactical bombers are powerful offensive assets Japan ignores too often.

FIGHTER (attack 3, move 4):
  - Can fly from Japan → Shantung → attack Kiangsi in one turn (4 zones range)
  - Must land on a FRIENDLY territory after combat (within remaining movement)
  - Pair with ground forces to push borderline attacks above 55% win rate
  - After combat, reposition to forward base (Manchuria, Kiangsu, Shantung)

TAC BOMBER (attack 3 solo / attack 4 when paired with fighter or tank):
  - Always pair with a fighter or an attacking tank for the +1 attack boost
  - Same movement rules as fighter

HOW TO USE AIRCRAFT IN COMBAT:
  1. Calculate ground force win rate (tool_predict_battle_odds)
  2. If win rate is 40-55%, add 1-2 fighters to push above 55%
  3. Move aircraft with tool_move_units in Combat Move phase
  4. Ensure safe landing territory exists BEFORE committing aircraft
  5. If no safe landing within range, do NOT send aircraft

=== TANK (armour) USAGE — CRITICAL ===
Tanks are Japan's highest-value ground unit.
  - 1 tank + 1 inf on a transport = optimal load (Att 4 total, 9 PU)
  - Use tanks to spearhead attacks on fortified territories
  - Move 2 means tanks can exploit breakthroughs (move to captured territory)
  - ALWAYS load 1 tank + 1 infantry per transport — never send a tank alone

=== TRANSPORT CYCLING — MANDATORY EVERY ROUND ===
Japan is an island. Troops on Japan = zero combat value.
Transports are the ONLY way to project force to the mainland.

NCM priority order:
  1. UNLOAD: For each transit SZ with ground units — land troops, then return transport to 6 Sea Zone.
  2. LOAD & DISPATCH: For each empty transport at 6 Sea Zone — load 1 armour + 1 infantry (best combo),
     dispatch toward the current frontline (use neighbors in get_state to pick the right SZ).
  3. Never leave an empty transport idle at 6 SZ when Japan has land units to ship.

=== MAP TOPOLOGY ===
Each territory in get_state includes a "neighbors" list — use it directly.

=== RELEVANT RULES & EXPERIENCE ===
{rag_context}

=== AVAILABLE TOOLS ===
- tool_get_state()                                   — observe the board (call first)
- tool_get_legal_actions()                           — see what actions are available
- tool_buy_units(items_json)                         — Purchase phase: buy units
- tool_move_units(from, to, units_json)              — move / attack; also used for landing and transport return
- tool_transport_units(load_from, load_sz, transit_sz, units_json) — NCM: load + dispatch transport
- tool_predict_battle_odds(attacker_json, defender_json) — Monte Carlo simulator (±1.5% accuracy)
- tool_end_turn()                                    — end current phase (REQUIRED after each phase)

=== ACTION PROTOCOL ===
1. Call tool_get_state to observe the board
2. Execute actions based on phase instructions
3. ALWAYS call tool_end_turn() to finish the phase — game will not advance otherwise
4. Explain each decision in English (briefly)
"""


# ─────────────────────────────────────────────────────────────
# build_agent()
#  GPT 4o + 工具 + Prompt
# rag_context 参数：Step 3 完成后传入检索到的规则和经验
# ─────────────────────────────────────────────────────────────

def build_agent(
    rag_context: str = "No experience available, using base rules.",
    handler: "QuietCallbackHandler | None" = None,
) -> tuple["AgentExecutor", "QuietCallbackHandler"]:
    """
    Build an AgentExecutor.
    Returns (executor, handler). Handler current_phase can be updated
    per phase in run_full_turn to control output color.
    """
    if handler is None:
        handler = QuietCallbackHandler()

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)

    tools = [tool_get_state, tool_get_legal_actions,
             tool_predict_battle_odds,
             tool_end_turn, tool_buy_units, tool_place_units,
             tool_transport_units, tool_move_units]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT.format(rag_context=rag_context)),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        callbacks=[handler],
        max_iterations=25,
    )
    return executor, handler


# ─────────────────────────────────────────────────────────────
# 阶段指令表
# 每个 stepName 对应一段具体指令，告诉 GPT 这个阶段该干什么
# ─────────────────────────────────────────────────────────────

PHASE_INSTRUCTIONS = {
    "japanesePolitics": """
=== POLITICS PHASE ===
Optionally declare war on a power, or do nothing.
Warning: declaring war on UK also triggers war with ANZAC; any western declaration accelerates US entry.
If you do not wish to declare war, call tool_end_turn() immediately.

► STOP: After tool_end_turn() returns ok=true, output "Politics phase complete." and stop ALL tools.
""",

    "japanesePurchase": """
=== PURCHASE PHASE === Budget: max 3 tool calls.

MANDATORY: Follow the PURCHASE ADVISOR above exactly. It has already calculated the optimal buy.
  - If it says buy transports → you MUST buy that many transports FIRST.
  - Remaining PUs → buy armour + infantry pairs (9 PU each: 1 armour + 1 infantry).
  - Leftover PUs that can't afford a pair → buy infantry (3 PU each).
  - Spend ALL PUs (leave at most 2 unspent).

Step 1: call tool_get_state once (verify PUs).
Step 2: call tool_buy_units with EXACTLY what the PURCHASE ADVISOR recommended.
Step 3: call tool_end_turn().

► STOP: After tool_end_turn() returns ok=true, output your purchase summary and stop ALL tools.
""",

    "japaneseCombatMove": """
=== COMBAT MOVE PHASE ===

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE GUARD — LEGAL ACTIONS IN COMBAT MOVE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ ALLOWED: Moving units INTO enemy-owned territories (attack).
✓ ALLOWED: Unloading transports onto ENEMY territories (amphibious attack).
✗ FORBIDDEN: Moving units to friendly/Japanese territories — this is NOT repositioning phase.
✗ FORBIDDEN: Unloading transports onto FRIENDLY territories — defer to Noncombat Move.
✗ FORBIDDEN: Retrying the same action after it fails — skip immediately and move on.

⚠ COMMITTED UNITS RULE (CRITICAL — prevents infinite loops):
  Units marked "⚠COMMITTED" in the state have ALREADY MOVED this phase.
  They have ZERO remaining movement points and CANNOT move again.
  Do NOT include COMMITTED units in any attack plan or tool_move_units call.
  If a tool_move_units call returns ok=false, the units may have already been moved.
  After ANY failed move: SKIP that action entirely and move to the next target.
  NEVER retry a failed move with the same or different destination — the units are spent.

⚠ CRITICAL CHECK BEFORE EVERY tool_move_units CALL:
  Look up the destination territory owner in the game state.
  If owner == "Japanese" or owner == "Neutral" → this move is ILLEGAL in Combat Move.
  Only proceed if owner == "Chinese", "British", "Americans", or another enemy nation.

AMPHIBIOUS RULE:
  - Before declaring an amphibious attack, verify the sea zone is FREE of enemy warships.
  - If the landing is rejected and the destination territory is FRIENDLY, defer it to NCM phase.
  - If war restrictions block a territory (not at war), mark it cancelled and do NOT retry.

POLITICAL RESTRICTIONS:
  - You cannot enter UK, US, ANZAC, or Chinese territories without being at war.
  - If an action is blocked by politics, cancel it immediately. Do not retry.

IRON RULE: You MUST evaluate ALL adjacent enemy territories (not just 1-2).
If tool_move_units is not called, the system will force a re-evaluation.
MULTI-TARGET: Attack every target where ratio ≥ 1.5 — do NOT stop after one attack.
Typical combat rounds should produce 2-4 attacks across multiple fronts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FREE CAPTURE RULE (check FIRST, before any force ratio evaluation):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scan ALL enemy territories adjacent to Japanese forces.
If ANY enemy territory has 0 defenders (empty):
  → Capture it IMMEDIATELY with 1 infantry from the nearest adjacent territory.
  → Free IPC should NEVER be left on the table.
  → No battle odds needed — 0 defenders = guaranteed capture.
  → Do this BEFORE evaluating other attacks.
Example: Yunnan has 0 Chinese infantry → move 1 inf from Kwangsi to Yunnan. Done. +2 IPC.

Tool budget:
  tool_get_state           × 1
  tool_predict_battle_odds × 0-6 (evaluate every borderline target)
  tool_move_units          × 0-8 (ground + air moves for MULTIPLE targets)
  tool_end_turn            × 1
  ─────────── total max 16 calls ───────────

Step 1 (required, once only): tool_get_state
  Record: unit counts in all territories. Do not call again.

Step 2: evaluate ALL adjacent enemy territories. Decision rules:

  FORCE RATIO + SIMULATOR:
    Ratio ≥ 1.5  → attack directly with tool_move_units (no simulator needed)
    Ratio 1.0–1.5 → call tool_predict_battle_odds for exact win rate:
                    win rate ≥ 55%  → attack
                    win rate < 55%  → cancel, consolidate in NCM
    Ratio < 1.0   → cancel, consolidate forces in NCM phase

  AIRCRAFT SUPPORT (use fighters and tac bombers to push borderline attacks):
    If ground win rate is 40-55%, check if adding 1-2 fighters raises it above 55%.
    Move fighters with tool_move_units (same as ground attack move).
    CRITICAL: verify fighters have a safe landing territory within range BEFORE attacking.
    Tac bombers: always pair with a fighter or an attacking tank (+1 attack bonus).

  PRIORITY: Follow the active Strategic Plans (injected in Round Plan above).
  If no plans: capture adjacent enemy territories with highest IPC value first.

  NOTES:
    - Move ground units AND aircraft in this phase. Aircraft must land after battle.
    - Skip failed attacks immediately, do not retry the same target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY PRE-END-TURN CHECKLIST — complete ALL before tool_end_turn():
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 1. Every adjacent enemy territory has been evaluated (force ratio or battle odds call).
□ 2. An explicit GO / NO-GO decision has been recorded for EACH candidate target.
□ 3. The NONCOMBAT PLAN below has been designed (which territory to stage forces toward next round).

If ANY checkbox is not checked, DO NOT call tool_end_turn(). Continue evaluating.

Step 3 (required): Before calling tool_end_turn(), write BOTH plans below IN YOUR REASONING TEXT.
  The plans must appear in your message content BEFORE the tool_end_turn() call.
  tool_end_turn() terminates execution immediately — anything written after it is lost.

Write the following plans NOW (before calling tool_end_turn):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMBAT PLAN — Round N
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategic Direction: [one sentence strategic assessment]
Primary Objective: [highest priority territory THIS round]

Confirmed Attacks:
1. [Territory] — Win rate: X% — Attacking with: [forces] — Reason: [why]
(write None if no attacks)

Cancelled Attacks:
- [Territory] — Reason: [win rate too low / insufficient forces / off strategy]
(write None if none cancelled)

Units held back for defense:
- [units] staying in [territory] because [reason]
(write None if none held back)

Reflexion Reference: [lesson referenced, or None]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NONCOMBAT PLAN — Round N
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next Round's Primary Target: [territory you plan to attack NEXT round]
Staging Territory: [YOUR friendly territory adjacent to that target]
Defender Estimate: [approx defender unit count in the target]
Forces Needed: [units and count required for ≥1.5 ratio or ≥55% win rate]

Reinforcement Moves This NCM:
1. Move [X units] from [Origin] to [Staging Territory] — Why: [pushes toward target]
(write None if no land reinforcement is needed this round)

Then call tool_end_turn() once. Execution stops automatically after that.
""",

    "japaneseBattle": """
=== BATTLE PHASE ===
Combat is resolved automatically by the game. No action required.
Call tool_end_turn() immediately.

► STOP: After tool_end_turn() returns ok=true, output "Battle phase complete." and stop ALL tools.
""",

    "japaneseNonCombatMove": """
=== NONCOMBAT MOVE PHASE === Budget: max 20 tool calls. Last call MUST be tool_end_turn().

RULES: Move only between FRIENDLY territories. No attacks. If a tool returns ok=false, skip it.
TOOLS: tool_transport_units to load+dispatch, tool_move_units to unload/reposition.

Step 1 (required): call tool_get_state once. Note:
  A) Sea zones with Japanese ground units (troops in transit — need landing)
  B) Empty transports in 6 Sea Zone (ready to load)
  C) Land units on Japan island (waiting for transport)
  D) Rear areas (Manchuria, Korea) with idle units

EXECUTION ORDER:
  1. UNLOAD: For each SZ with ground units → land troops to adjacent friendly territory,
     then return transport to 6 Sea Zone.
  2. REAR AUDIT: Move idle units from Manchuria/Korea toward the frontline (1 hop per round).
     Keep only 1 infantry as garrison.
  3. LOAD & DISPATCH: For each empty transport in 6 SZ → load 1 armour + 1 infantry (optimal),
     pick transit SZ near current frontline, call tool_transport_units per transport.
  4. AIRCRAFT: Reposition fighters/tac bombers to forward bases within range.
  5. CONSOLIDATION: Stage ground forces adjacent to next round's attack target
     (follow the NONCOMBAT PLAN from Combat Move if available).

Step N (required): call tool_end_turn().

► STOP: After tool_end_turn() returns ok=true, output your NCM summary and stop ALL tools.
""",

    "japanesePlace": """
=== PLACE UNITS PHASE ===
Place Units is handled automatically. Call tool_end_turn() only.
""",

    "japaneseEndTurn": """
=== COLLECT INCOME PHASE ===
Income is collected automatically. No action needed.
Call tool_end_turn() to complete the round.

► STOP: After tool_end_turn() returns ok=true, output "Round complete." and stop ALL tools.
""",
}

DEFAULT_PHASE_INSTRUCTION = """
Unknown phase. Call tool_get_state and tool_get_legal_actions to assess the situation,
then execute appropriate actions based on legal_actions, and call tool_end_turn() when done.
"""


def get_phase_instruction(step_name: str) -> str:
    return PHASE_INSTRUCTIONS.get(step_name, DEFAULT_PHASE_INSTRUCTION)


# ─────────────────────────────────────────────────────────────
# 限流重试包装
# ─────────────────────────────────────────────────────────────

_MAX_ITER_SIGNALS = ("stopped due to max iterations", "agent stopped due to max")


def _invoke_with_retry(
    agent: "AgentExecutor",
    instruction: str,
    max_retries: int = 5,
) -> str:
    """
    Invoke agent with error handling:
    1. 429 rate limit → exponential backoff (5s/10s/20s/40s), up to max_retries
    2. max iterations → retry once with a compact instruction
    Other exceptions → return error string
    """
    import time as _time

    for attempt in range(max_retries):
        try:
            result = agent.invoke({"input": instruction})
            output = result.get("output", "")

            if any(sig in output.lower() for sig in _MAX_ITER_SIGNALS):
                print(
                    f"  {Colors.YELLOW}[max-iter] Step limit reached, retrying with compact instruction...{Colors.RESET}"
                )
                compact = (
                    "Previous attempt was cut off due to step limit.\n"
                    "Complete the single most important action (or skip), then immediately call tool_end_turn().\n"
                    f"Phase summary: {instruction.strip()[:150]}…"
                )
                try:
                    r2 = agent.invoke({"input": compact})
                    return r2.get("output", output)
                except Exception:
                    pass
            return output

        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "429" in err
                or "rate_limit_exceeded" in err
                or "tokens per min" in err
                or "Rate limit" in err
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(
                    f"  {Colors.YELLOW}[Rate Limit 429] Waiting {wait}s before retry"
                    f" ({attempt + 1}/{max_retries - 1})...{Colors.RESET}"
                )
                _time.sleep(wait)
                instruction = (
                    "IMPORTANT: Previous attempt was interrupted by rate limit BEFORE any action could complete.\n"
                    "NO actions have been executed yet — you MUST still perform ALL required actions for this phase.\n"
                    "For Purchase phase: you MUST buy units before calling tool_end_turn. Do NOT skip purchasing.\n"
                    "For Combat Move: you MUST evaluate targets and attack before tool_end_turn.\n"
                    "For Noncombat Move: you MUST execute transport operations before tool_end_turn.\n"
                    "Start by calling tool_get_state, then execute the full phase instruction below.\n\n"
                    + instruction
                )
                continue
            if is_rate_limit:
                print(f"  {Colors.RED}[Rate Limit] Exhausted {max_retries} retries — falling back{Colors.RESET}")
            return f"Agent error: {err}"
    return "Unknown error"


# ─────────────────────────────────────────────────────────────
# Strategic Plan Initialization (game start, 1 LLM call)
# ─────────────────────────────────────────────────────────────

from memory import (
    StrategicPlan, StrategicPlansInit,
    load_national_strategy, ns_to_prompt_text,
)

_INIT_PLANS_PROMPT = """You are the supreme strategist for {nation} in Axis & Allies Pacific 1940.
Round 1 is about to begin.

=== NATIONAL STRATEGY (from previous games) ===
{ns_text}

=== CURRENT BOARD STATE ===
{compressed_state}

{nation} PUs: {pus}

Based on the national strategy and the current board state, create 2-4 Strategic Plans
for this game. Each plan should:
1. Have a clear, specific objective (territory to capture, defensive line to hold, etc.)
2. Include the REASON why this plan matters strategically
3. List concrete ACTIONS needed (which units move where, what to purchase)
4. State the EXPECTED OUTCOME with territory names and a target round
5. Be ordered by priority (plan 1 = most urgent)

If the national strategy already has validated plans, adopt them with adjustments based
on the current board state. If it has failed plans, learn from the lessons_learned.

Plan IDs should be short descriptive strings like "sp_secure_coast", "sp_capture_fic".
"""


def _initialize_strategic_plans(
    ns: dict,
    round_num: int = 1,
) -> list[StrategicPlan]:
    """
    Game start: generate initial Strategic Plans from National Strategy + board state.
    Called once before Round 1.
    """
    state = _client.get_state()
    compressed = _compress_state_for_llm(state)
    pus = state.get("japan", {}).get("pus", "?")
    nation = ns.get("nation", "Japan")
    ns_text = ns_to_prompt_text(ns) or "(No prior strategy — first game)"

    prompt = _INIT_PLANS_PROMPT.format(
        nation=nation,
        ns_text=ns_text,
        compressed_state=compressed,
        pus=pus,
    )

    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        structured = llm.with_structured_output(StrategicPlansInit)
        result = structured.invoke(prompt)
        plans = result.plans

        print(f"\n{Colors.CYAN}{'━'*60}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}  STRATEGIC PLANS INITIALIZED — {len(plans)} plans{Colors.RESET}")
        print(f"{Colors.CYAN}{'━'*60}{Colors.RESET}")
        for i, p in enumerate(plans, 1):
            print(f"  {Colors.CYAN}{i}. [{p.plan_id}] {p.name} (target round {p.target_round}){Colors.RESET}")
            print(f"     {Colors.DIM}Reason: {p.reason}{Colors.RESET}")
            print(f"     {Colors.DIM}Expected: {p.expected_outcome}{Colors.RESET}")
        print(f"{Colors.CYAN}{'━'*60}{Colors.RESET}\n")

        return plans
    except Exception as e:
        print(f"  {Colors.RED}[Init Plans] Failed: {e}{Colors.RESET}")
        # Fallback: create plans from NS file directly
        fallback = []
        for sp in ns.get("strategic_plans", []):
            fallback.append(StrategicPlan(
                plan_id=sp["id"],
                name=sp["name"],
                reason=sp.get("reason", ""),
                actions=sp.get("key_actions", []),
                expected_outcome=sp.get("expected_outcome", ""),
                target_round=int(sp.get("target_rounds", "3").split("-")[-1]),
                status="active",
                progress=[],
            ))
        if fallback:
            print(f"  {Colors.DIM}[Init Plans] Fallback: loaded {len(fallback)} plans from NS file{Colors.RESET}")
        return fallback


# ─────────────────────────────────────────────────────────────
# Layer 2 — Round Plan: unified plan generated before phase loop
# ─────────────────────────────────────────────────────────────

_ROUND_PLAN_PROMPT = """You are planning a turn in Axis & Allies Pacific 1940.
Round: {round_num} | Stage: {stage}
PUs available: {pus}

=== ACTIVE STRATEGIC PLANS ===
{plans_section}

=== BOARD STATE ===
{compressed_state}

{prev_rounds_section}

{rag_section}

Your job: create a ROUND PLAN that advances the active Strategic Plans.

For each active plan, state:
  - What progress can be made THIS round
  - Which specific attacks/moves serve this plan

Then generate the full execution plan using this 3-step chain of thought:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — TARGET EVALUATION (linked to Strategic Plans)
  For each active plan, list candidate targets this round.
  For each candidate: owner, defender units, adjacent friendly units, force ratio.

STEP 2 — ATTACK DECISIONS
  For each candidate target:
    ratio >= 1.5 → GO (state which plan this serves)
    ratio 1.0-1.5 → BORDERLINE — check aircraft support
    ratio < 1.0 or not at war → NO-GO (stage toward it for next round)

  THIS ROUND ATTACKS: [targets with units and origins, linked to plan IDs]
  NEXT ROUND STAGING: [targets to prepare for, staging territory, units needed]

STEP 3 — NONCOMBAT MOVE + PURCHASE PLAN
  Noncombat moves (priority order):
    1. Unload loaded transports at frontline
    2. Return empty transports to home loading zone
    3. Ground force redeployment toward staging territories
    4. Load & dispatch from home loading zone

  Purchase plan (budget: {pus} PUs):
    - Transport check: if land units at home ≥ 4 and transports ≤ 2 → buy transports
    - Units needed for NEXT ROUND targets
    - Backfill territories thinned by attacks
    - Placement: which factory gets which units

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Be concrete — territory names, unit counts, force ratios. No vague statements.
"""


def _generate_round_plan(
    round_num: int,
    plan_tracker: list[StrategicPlan],
    prev_round_summaries: list[str] | None = None,
    rag_context: str = "",
) -> str:
    """
    Layer 2: Generate a unified Round Plan that advances active Strategic Plans.
    One LLM call that coordinates Purchase/Combat/NCM, referencing the plan tracker.
    """
    state = _client.get_state()
    compressed = _compress_state_for_llm(state)
    pus = state.get("japan", {}).get("pus", "?")

    stage = "Early" if round_num <= 3 else ("Mid" if round_num <= 6 else "Late")

    # Build plans section from active plans
    plans_lines = []
    for p in plan_tracker:
        if p.status != "active":
            continue
        progress_str = "; ".join(p.progress[-3:]) if p.progress else "(no progress yet)"
        plans_lines.append(
            f"[{p.plan_id}] {p.name} (target round {p.target_round})\n"
            f"  Reason: {p.reason}\n"
            f"  Actions: {'; '.join(p.actions)}\n"
            f"  Expected: {p.expected_outcome}\n"
            f"  Progress: {progress_str}"
        )
    plans_section = "\n\n".join(plans_lines) if plans_lines else "(no active plans)"

    prev_section = ""
    if prev_round_summaries:
        prev_text = "\n".join(prev_round_summaries[-3:])
        prev_section = f"Previous rounds:\n{prev_text}"

    rag_section = ""
    if rag_context and rag_context.strip():
        rag_section = f"Relevant experience from past games:\n{rag_context.strip()}"

    prompt = _ROUND_PLAN_PROMPT.format(
        round_num=round_num,
        stage=stage,
        pus=pus,
        plans_section=plans_section,
        compressed_state=compressed,
        prev_rounds_section=prev_section,
        rag_section=rag_section,
    )

    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        response = llm.invoke(prompt)
        plan_text = response.content.strip()
        print(f"\n{Colors.CYAN}{'━'*60}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}  ROUND PLAN — Round {round_num}{Colors.RESET}")
        print(f"{Colors.CYAN}{'━'*60}{Colors.RESET}")
        for line in plan_text.split("\n"):
            print(f"  {Colors.CYAN}{line}{Colors.RESET}")
        print(f"{Colors.CYAN}{'━'*60}{Colors.RESET}\n")
        return plan_text
    except Exception as e:
        print(f"  {Colors.RED}[Round Plan] Generation failed: {e}{Colors.RESET}")
        return ""


# ─────────────────────────────────────────────────────────────
# Round Summary: compress a round's game_log into one paragraph
# ─────────────────────────────────────────────────────────────

def _summarize_round(round_num: int, game_log: list[str]) -> str:
    """
    Layer 2 support: compress one round's game_log into a 2-3 line summary.
    Uses a cheap model to minimize cost. Fed into next round's plan generation.
    """
    if not game_log:
        return f"Round {round_num}: No actions recorded."

    log_text = "\n".join(game_log[-20:])
    prompt = (
        f"Summarize Japan's Round {round_num} in Axis & Allies Pacific 1940.\n"
        f"Game log:\n{log_text}\n\n"
        f"Write exactly 2-3 sentences. Include: territories attacked/captured, "
        f"units purchased, key troop movements, and IPC changes. Be specific with names and numbers."
    )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(prompt)
        summary = f"Round {round_num}: {response.content.strip()}"
        print(f"  {Colors.DIM}[Round Summary] {summary}{Colors.RESET}")
        return summary
    except Exception as e:
        print(f"  {Colors.DIM}[Round Summary] Failed: {e}{Colors.RESET}")
        return f"Round {round_num}: (summary unavailable)"


# ─────────────────────────────────────────────────────────────
# Layer 3 — Strategic Reassessment (every 2 rounds)
# ─────────────────────────────────────────────────────────────

_STRATEGY_REASSESS_PROMPT = """You are the supreme strategist in Axis & Allies Pacific 1940.
Round: {round_num}

Board state:
{compressed_state}

History:
{round_summaries}

Current Strategic Plans:
{plans_text}

Evaluate each active plan:
1. Is it ON TRACK, BEHIND SCHEDULE, or BLOCKED?
2. Should it be CONTINUED, MODIFIED, or ABANDONED?
3. Are there NEW opportunities visible on the board that warrant a new plan?

For each plan, output:
  [plan_id] STATUS: ON TRACK / BEHIND / BLOCKED
  [plan_id] ACTION: CONTINUE / MODIFY: <what to change> / ABANDON: <reason>

If you want to ADD a new plan, output:
  NEW PLAN: <name> | reason: <why> | actions: <what to do> | target_round: <N>

Be concrete — reference territory names, unit counts, force ratios.
"""


def _reassess_strategy(
    round_num: int,
    plan_tracker: list[StrategicPlan],
    round_summaries: list[str],
) -> list[StrategicPlan]:
    """
    Layer 3: Reassess strategic plans every 2 rounds.
    Can modify, abandon, or add plans. Returns updated plan list.
    """
    state = _client.get_state()
    compressed = _compress_state_for_llm(state)
    summaries_text = "\n".join(round_summaries) if round_summaries else "(first assessment)"

    plans_lines = []
    for p in plan_tracker:
        if p.status != "active":
            continue
        progress_str = "; ".join(p.progress[-3:]) if p.progress else "(no progress yet)"
        plans_lines.append(
            f"  [{p.plan_id}] {p.name} — target round {p.target_round}\n"
            f"    Progress: {progress_str}\n"
            f"    Expected: {p.expected_outcome}"
        )
    plans_text = "\n".join(plans_lines) if plans_lines else "(no active plans)"

    prompt = _STRATEGY_REASSESS_PROMPT.format(
        round_num=round_num,
        compressed_state=compressed,
        round_summaries=summaries_text,
        plans_text=plans_text,
    )

    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        response = llm.invoke(prompt)
        text = response.content.strip()

        print(f"\n{Colors.MAGENTA}{'━'*60}{Colors.RESET}")
        print(f"{Colors.MAGENTA}{Colors.BOLD}  STRATEGIC REASSESSMENT — Round {round_num}{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'━'*60}{Colors.RESET}")
        for line in text.split("\n"):
            print(f"  {Colors.MAGENTA}{line}{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'━'*60}{Colors.RESET}\n")

        # Parse abandon directives
        for line in text.split("\n"):
            stripped = line.strip()
            for p in plan_tracker:
                if p.plan_id in stripped and "ABANDON" in stripped.upper():
                    p.status = "abandoned"
                    print(f"  {Colors.MAGENTA}[Strategy] ABANDONED: {p.name}{Colors.RESET}")

        # Parse new plan directives
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("NEW PLAN:"):
                parts = stripped.split("|")
                name = parts[0].replace("NEW PLAN:", "").strip()
                plan_id = f"sp_r{round_num}_{name[:10].lower().replace(' ', '_')}"
                reason = ""
                actions_str = ""
                target = round_num + 3
                for part in parts[1:]:
                    kv = part.strip()
                    if kv.lower().startswith("reason:"):
                        reason = kv.split(":", 1)[1].strip()
                    elif kv.lower().startswith("actions:"):
                        actions_str = kv.split(":", 1)[1].strip()
                    elif kv.lower().startswith("target_round:"):
                        try:
                            target = int(kv.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                new_plan = StrategicPlan(
                    plan_id=plan_id,
                    name=name,
                    reason=reason,
                    actions=[a.strip() for a in actions_str.split(";") if a.strip()],
                    expected_outcome=f"(added at round {round_num} reassessment)",
                    target_round=target,
                    status="active",
                    progress=[],
                )
                plan_tracker.append(new_plan)
                print(f"  {Colors.MAGENTA}[Strategy] NEW PLAN: {name} (target round {target}){Colors.RESET}")

        return plan_tracker
    except Exception as e:
        print(f"  {Colors.RED}[Strategy] Reassessment failed: {e}{Colors.RESET}")
        return plan_tracker


# ─────────────────────────────────────────────────────────────
# Auto Place Units (programmatic — no LLM)
# ─────────────────────────────────────────────────────────────

_NAVAL_UNIT_TYPES = frozenset({
    "transport", "destroyer", "submarine", "cruiser", "carrier", "battleship",
})


def _auto_place_units() -> str:
    """
    Place all purchased units without LLM. Strategy:
      - Land/air units → first land territory in placeOptions (usually Japan)
      - Naval units    → first sea zone in placeOptions (usually 6 Sea Zone)
    Falls back to putting everything in the first available territory.
    """
    global _last_purchase
    if not _last_purchase:
        print(f"  {Colors.DIM}[Place] No purchase record — skipping{Colors.RESET}")
        _client.act_end_turn()
        return "No units to place."

    state = _client.get_state()
    place_options = state.get("placeOptions", [])
    if not place_options:
        print(f"  {Colors.DIM}[Place] placeOptions empty — skipping{Colors.RESET}")
        _last_purchase.clear()
        _client.act_end_turn()
        return "No placement slots available."

    land_territory = None
    sea_territory = None
    for opt in place_options:
        name = opt.get("territory", "")
        if "Sea Zone" in name and sea_territory is None:
            sea_territory = name
        elif "Sea Zone" not in name and land_territory is None:
            land_territory = name

    fallback = place_options[0].get("territory", "Japan")
    if land_territory is None:
        land_territory = fallback
    if sea_territory is None:
        sea_territory = fallback

    placements: list[dict] = []
    for item in _last_purchase:
        ut = item.get("unitType", "")
        count = item.get("count", 1)
        dest = sea_territory if ut in _NAVAL_UNIT_TYPES else land_territory
        placements.append({"territory": dest, "unitType": ut, "count": count})

    result = _client.act({"type": "PLACE_UNITS", "placements": placements})
    summary_parts = [f"{p['count']}x {p['unitType']} → {p['territory']}" for p in placements]
    summary = ", ".join(summary_parts)

    if result.get("ok"):
        print(f"  {Colors.GREEN}[Place] Auto-placed: {summary}{Colors.RESET}")
    else:
        err = result.get("error", "unknown")
        print(f"  {Colors.RED}[Place] Placement failed ({err}), trying end_turn anyway{Colors.RESET}")

    _last_purchase.clear()
    _client.act_end_turn()
    return f"Auto-placed: {summary}"


# ─────────────────────────────────────────────────────────────
# Purchase Advisor (programmatic — injected before Purchase phase)
# ─────────────────────────────────────────────────────────────

_HEAVY_TYPES = frozenset({"armour", "artillery", "mech_infantry"})
_LAND_TYPES = frozenset({"infantry", "armour", "artillery", "mech_infantry"})


def _compute_purchase_advice(state: dict) -> str:
    """
    Human-style purchase logic:
      1. How many transports do I need to clear the Japan stockpile in ~2 rounds?
      2. Buy transports to reach that target.
      3. ALL remaining PU → tank+infantry pairs (9 PU each). No standalone infantry.
    """
    japan_info = state.get("japan", {})
    pus = japan_info.get("pus", 0)
    if pus <= 0:
        return ""

    # unitsByTerritory is flat: {"Japan": {"infantry": 5, "armour": 2}, "6 Sea Zone": {"transport": 3}}
    units_by_terr = state.get("unitsByTerritory", {})

    # Count ALL Japanese transports
    total_transports = 0
    for terr_name, unit_counts in units_by_terr.items():
        if "Sea Zone" in terr_name:
            total_transports += unit_counts.get("transport", 0)

    # Count land units on Japan island
    japan_units = units_by_terr.get("Japan", {})
    japan_land = sum(japan_units.get(ut, 0) for ut in _LAND_TYPES)
    japan_tanks = japan_units.get("armour", 0)

    # Step 1: How many transports needed?
    # Each transport ships 2 units/round. Target: clear stockpile in ~2 rounds.
    # target_fleet * 2 * 2 >= japan_land → target_fleet >= japan_land / 4
    target_fleet = max((japan_land + 3) // 4, 3)  # min 3 transports
    buy_transports = max(0, target_fleet - total_transports)
    # Don't spend ALL PU on transports — leave at least 18 PU (2 tank+inf pairs)
    max_transport_budget = max(pus - 18, 0)
    buy_transports = min(buy_transports, max_transport_budget // 7)
    transport_cost = buy_transports * 7

    # Step 2: Remaining PU → tank+infantry pairs ONLY (9 PU each)
    remaining = pus - transport_cost
    pairs = remaining // 9  # 1 armour (6) + 1 infantry (3)
    leftover = remaining - pairs * 9
    extra_inf = leftover // 3  # only if can't afford a pair

    lines = [
        f"Status: {japan_land} land units on Japan ({japan_tanks} tanks), "
        f"{total_transports} transports, {pus} PUs"
    ]

    if buy_transports > 0:
        lines.append(
            f"⚠ STOCKPILE: {japan_land} units on Japan, only {total_transports} transports "
            f"(can ship {total_transports * 2}/round, need {target_fleet * 2}/round)."
        )
        lines.append(
            f"→ BUY {buy_transports} transport(s) ({transport_cost} PU) + "
            f"{pairs} armour + {pairs} infantry ({pairs * 9} PU)"
            + (f" + {extra_inf} extra infantry ({extra_inf * 3} PU)" if extra_inf else "")
            + f" = {transport_cost + pairs * 9 + extra_inf * 3} PU total."
        )
    else:
        lines.append(
            f"→ BUY {pairs} armour + {pairs} infantry = {pairs * 9} PU "
            f"(1 armour + 1 infantry per transport, optimal load)."
        )
        if extra_inf:
            lines.append(f"  + {extra_inf} extra infantry with remaining {leftover} PU.")

    if japan_tanks >= 3:
        lines.append(
            f"⚠ TANK PRIORITY: {japan_tanks} tanks on Japan — ship tanks FIRST this NCM!"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# run_full_turn()
# 跑完日本的一整个回合（所有阶段），直到轮到下一个玩家为止
# ─────────────────────────────────────────────────────────────

def run_full_turn(
    rag_context: str = "",
    start_phase: str = "",
    round_num: int = 1,
    plan_tracker: list[StrategicPlan] | None = None,
    prev_round_summaries: list[str] | None = None,
) -> "list[str]":
    """
    Execute a complete Japan turn (or resume from a mid-turn phase).
    start_phase: logged phase name for display; actual execution follows get_phase().
    plan_tracker: active Strategic Plans (game-level, cross-round).
    prev_round_summaries: cross-round memory from previous rounds.
    """
    import time as _time

    handler = QuietCallbackHandler()
    agent, handler = build_agent(rag_context=rag_context, handler=handler)
    game_log = []
    completed_phases = []
    phase_log: dict = {}

    title = f"from {start_phase}" if start_phase else "full turn"
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Japan turn start [{title}]{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    # ── Layer 2: Generate Round Plan referencing active Strategic Plans ──
    _active_plans = plan_tracker or []
    round_plan = _generate_round_plan(
        round_num=round_num,
        plan_tracker=_active_plans,
        prev_round_summaries=prev_round_summaries,
        rag_context=rag_context,
    )

    _has_rag = bool(rag_context and rag_context.strip())
    if _has_rag:
        print(f"{Colors.GREEN}{'─'*60}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}  📚 Strategic Experience (injected into Round Plan){Colors.RESET}")
        print(f"{Colors.GREEN}{'─'*60}{Colors.RESET}")
        ctx_lines = rag_context.strip().split("\n")
        for line in ctx_lines[:20]:
            print(f"  {Colors.GREEN}{line}{Colors.RESET}")
        if len(ctx_lines) > 20:
            print(f"  {Colors.DIM}  ...({len(ctx_lines)} lines, truncated){Colors.RESET}")
        print(f"{Colors.GREEN}{'─'*60}{Colors.RESET}\n")
    else:
        print(f"  {Colors.DIM}[RAG] No experience available — using base strategy{Colors.RESET}\n")

    # 日本一回合的标准阶段顺序，最后一个阶段完成 = 本轮结束
    _ROUND_END_PHASE = "japaneseEndTurn"

    while _client.is_our_turn():
        step_name = _client.get_phase()
        print(f"  {Colors.DIM}[Loop] get_phase() → {step_name} | completed: {completed_phases}{Colors.RESET}")

        if step_name == "unknown":
            _time.sleep(2)
            continue

        if step_name in completed_phases:
            # ── 关键判断：这个阶段已经做过了 ──
            # 情况 A：已完成整轮（endTurn做过）→ 新回合已开始，立即退出
            # 情况 B：还在同一回合的阶段切换间隙 → 等待游戏推进到下一阶段
            if _ROUND_END_PHASE in completed_phases:
                print(
                    f"\n  {Colors.BOLD}[New round detected] {step_name} reappeared after round completion — "
                    f"exiting run_full_turn to main loop{Colors.RESET}"
                )
                break
            # 同回合阶段切换中，短暂等待
            _time.sleep(1)
            continue

        # Update handler phase (controls output color) and reset move tracking
        handler.current_phase = step_name
        handler.reset_phase_tracking()
        _state_cache.reset()
        global _transports_used_this_phase
        _transports_used_this_phase = 0

        # ── Phase banner (display.py) ─────────────────────────────
        _pnum  = _PHASE_NUMBER.get(step_name, 0)
        _pname = _PHASE_DISPLAY_NAME.get(step_name, step_name)
        print_phase_header(_pnum, _pname)

        instruction = get_phase_instruction(step_name)

        # ── Inject Round Plan into every action phase ──
        if round_plan and step_name in (
            "japanesePurchase", "japaneseCombatMove", "japaneseNonCombatMove", "japanesePlace",
        ):
            instruction = (
                f"[ROUND PLAN — follow this coordinated plan]\n"
                f"{round_plan}\n"
                f"{'─'*40}\n\n"
                + instruction
            )

        # ── Purchase advice injection: match transport capacity ──
        if step_name == "japanesePurchase":
            try:
                _pstate = _client.get_state()
                _advice = _compute_purchase_advice(_pstate)
                if _advice:
                    print(f"  {Colors.CYAN}[Purchase Advisor] {_advice}{Colors.RESET}")
                    instruction = (
                        f"╔══ PURCHASE ORDER (MANDATORY — buy EXACTLY this) ══╗\n"
                        f"{_advice}\n"
                        f"╚══════════════════════════════════════════════════════╝\n\n"
                        + instruction
                    )
            except Exception as _pe:
                print(f"  {Colors.DIM}[Purchase] Advisor error: {_pe}{Colors.RESET}")

        # ── Transport capacity injection for Combat Move (Bug 3) ──
        if step_name == "japaneseCombatMove":
            try:
                _raw_state = _client.get_state()
                _cap_text = compute_transport_capacity(_raw_state)
                print_transport_capacity(_cap_text)
                instruction = (
                    f"TRANSPORT CAPACITY THIS ROUND:\n{_cap_text}\n\n"
                    f"TRANSPORT UTILIZATION RULE: Every transport must be assigned a task "
                    f"this round — either loading troops for an attack, or repositioning to "
                    f"be closer to the frontline for next round. A transport that ends the "
                    f"round completely idle (no move, no load) is wasted capacity.\n"
                    f"For each transport listed above, your COMBAT PLAN must state what it "
                    f"carries and where it is going.\n\n"
                    + instruction
                )
            except Exception as _te:
                print(f"  {Colors.DIM}[Transport] Could not compute capacity: {_te}{Colors.RESET}")

        # ── Carry NONCOMBAT PLAN from Combat Move into NCM ───────
        if step_name == "japaneseNonCombatMove":
            _ncmp = phase_log.get("noncombat_plan", "")
            _cp   = phase_log.get("combat_plan", "")
            if _ncmp:
                print_plan(_ncmp, label="Noncombat Plan (from Combat Move)")
                instruction = (
                    f"NONCOMBAT PLAN DESIGNED IN COMBAT MOVE — EXECUTE THIS FIRST:\n"
                    f"{_ncmp}\n\n{'─'*40}\n\n"
                    + instruction
                )
            elif _cp:
                # Fallback: carry combat plan if NCM plan wasn't extracted
                print_plan(_cp, label="Combat Plan reference (for NCM staging)")
                instruction = (
                    f"REFERENCE — COMBAT PLAN FROM PREVIOUS PHASE:\n"
                    f"{_cp}\n\n{'─'*40}\n\n"
                    + instruction
                )

        # ── Battle phase: loop end_turn until all battles are resolved ──
        if step_name == "japaneseBattle":
            _battle_count = 0
            while True:
                _battle_count += 1
                print(f"  {Colors.DIM}[Battle] Resolving battle #{_battle_count}...{Colors.RESET}")
                _client.act_end_turn()
                _time.sleep(0.5)
                _next_phase = _client.get_phase()
                if _next_phase != "japaneseBattle":
                    print(f"  {Colors.DIM}[Battle] All {_battle_count} battle(s) resolved → {_next_phase}{Colors.RESET}")
                    break
                if _battle_count >= 15:
                    print(f"  {Colors.RED}[Battle] Safety limit: {_battle_count} end_turn calls, forcing advance{Colors.RESET}")
                    break
            output = f"Battle phase complete ({_battle_count} battle(s) resolved)."
            log_entry = f"[{step_name}] {output}"
            game_log.append(log_entry)
            completed_phases.append(step_name)
            continue

        # ── Place Units: programmatic — no LLM needed ──
        if step_name == "japanesePlace":
            output = _auto_place_units()
            # Poll for phase advance (same as normal phases)
            _place_advanced = False
            for _poll in range(10):
                _time.sleep(0.3)
                try:
                    _next = _client.get_phase()
                except Exception:
                    _next = step_name
                if _next != step_name:
                    _place_advanced = True
                    break
            if not _place_advanced and _client.is_our_turn():
                print(f"  {Colors.RED}[Place] Phase did not advance — forcing END_TURN{Colors.RESET}")
                _client.act_end_turn()
                _time.sleep(1)
            log_entry = f"[{step_name}] {output}"
            game_log.append(log_entry)
            completed_phases.append(step_name)
            continue

        output = _invoke_with_retry(agent, instruction)

        # ── Extract combat + noncombat plans after Combat Move ───
        # With return_direct=True, `output` is the tool return string, not the LLM plan.
        # Use handler.last_llm_output (LLM reasoning before tool_end_turn) for plan extraction.
        if step_name == "japaneseCombatMove":
            _plan_src = handler.last_llm_output if handler.last_llm_output.strip() else output
            _cp = _extract_combat_plan(_plan_src)
            if _cp:
                phase_log["combat_plan"] = _cp
            _ncmp = _extract_noncombat_plan(_plan_src)
            if _ncmp:
                phase_log["noncombat_plan"] = _ncmp
        if output.startswith("Agent error"):
            print(f"  {Colors.RED}[Warning] {output}{Colors.RESET}")

        if step_name == "japaneseCombatMove" and not handler.move_tools_called:
            # CRITICAL: check whether the game has already advanced past Combat Move.
            # The agent may have called tool_end_turn() without attacking, advancing
            # the game to Battle/NCM. Retrying in the wrong phase would corrupt the turn.
            _time.sleep(0.5)  # brief pause so game state settles
            _current_game_phase = _client.get_phase()
            if _current_game_phase != "japaneseCombatMove":
                print(
                    f"\n  {Colors.RED}[Warning] No attacks in Combat Move, but game already advanced to "
                    f"'{_current_game_phase}'. Skipping retry to avoid phase corruption.{Colors.RESET}"
                )
            else:
                # Phase is still Combat Move — safe to retry
                print(f"\n  {Colors.RED}[Warning] Combat Move: no attacks detected! Forcing re-evaluation...{Colors.RESET}")
                retry_combat = (
                    "You completed Combat Move without calling tool_move_units at all.\n"
                    "This is not allowed. Complete the mandatory checklist now:\n\n"
                    "Step 1: tool_get_state — record all territory unit counts\n"
                    "Step 2: For each adjacent enemy territory, compute force ratio:\n"
                    "  ratio ≥ 1.5  → attack with tool_move_units\n"
                    "  ratio 1.0-1.5 → call tool_predict_battle_odds; attack if ≥55%\n"
                    "  ratio < 1.0  → cancel, note in Combat Plan\n"
                    "Step 3: Design NONCOMBAT PLAN for staging forces toward next round's target\n"
                    "Step 4: tool_end_turn(), then output COMBAT PLAN + NONCOMBAT PLAN\n"
                )
                output2 = _invoke_with_retry(agent, retry_combat, max_retries=2)
                if output2 and not output2.startswith("Agent error"):
                    output = output2
                    print(f"  {Colors.WHITE}[Retry] Combat Move re-evaluation complete{Colors.RESET}")

        # Wait for phase to advance, polling quickly to avoid missing fast transitions (e.g. NCM after Battle)
        for _poll in range(10):
            _time.sleep(0.2)
            try:
                _next = _client.get_phase()
            except Exception:
                _next = step_name
            if _next != step_name:
                break
        else:
            if _client.is_our_turn():
                print(f"  {Colors.RED}[Fallback] Phase {step_name} not advanced — forcing END_TURN{Colors.RESET}")
                _client.act_end_turn()

        log_entry = f"[{step_name}] {output}"
        game_log.append(log_entry)
        completed_phases.append(step_name)

    # ── 循环退出后的最终兜底 ──
    # is_our_turn() 可能因短暂网络抖动返回 False 导致提前退出。
    # 如果 japaneseEndTurn 还没被处理，强制执行一次，防止游戏卡死。
    if _ROUND_END_PHASE not in completed_phases:
        _time.sleep(2)
        try:
            if _client.is_our_turn():
                remaining = _client.get_phase()
                print(
                    f"\n  {Colors.RED}[Final Fallback] Loop exited early, {_ROUND_END_PHASE} not done, "
                    f"current phase={remaining} — forcing END_TURN...{Colors.RESET}"
                )
                _client.act_end_turn()
        except Exception:
            pass

    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Japan turn complete{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    return game_log


# ─────────────────────────────────────────────────────────────
# 直接运行此文件时：快速测试 Agent 能否跑完一个完整回合
# 用法：python agent.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Agent test (game must be running) ===\n")
    logs = run_full_turn()
    print("\nTurn action log:")
    for entry in logs:
        print(" ", entry)
