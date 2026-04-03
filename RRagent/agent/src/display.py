"""
display.py — Terminal output helpers for the Japan agent.

Standalone module; no imports from agent.py or any project file.
All color output degrades gracefully when stdout is not a TTY.
"""
from __future__ import annotations

import sys
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# ANSI color codes
# ─────────────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
ORANGE = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
MAGENTA = "\033[95m"


def _use_color() -> bool:
    """Return True only when stdout is a real TTY that likely supports ANSI."""
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _c(code: str) -> str:
    """Return ANSI code only when color is supported; empty string otherwise."""
    return code if _use_color() else ""


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_phase_header(phase_number: int, phase_name: str) -> None:
    """Print a boxed phase banner, e.g.
    ╔══════════════════════════════════════════╗
    ║        PHASE 2 — PURCHASE UNITS          ║
    ╚══════════════════════════════════════════╝
    """
    title = f"PHASE {phase_number} — {phase_name.upper()}"
    width = max(44, len(title) + 4)
    padded = title.center(width)
    border = "═" * width
    try:
        print(f"\n{_c(BOLD)}╔{border}╗{_c(RESET)}")
        print(f"{_c(BOLD)}║{padded}║{_c(RESET)}")
        print(f"{_c(BOLD)}╚{border}╝{_c(RESET)}")
    except UnicodeEncodeError:
        print(f"\n{'='*(width+2)}")
        print(f"  {title}")
        print(f"{'='*(width+2)}")


def print_rag_context(docs: List[str]) -> None:
    """Print up to 3 RAG results numbered ①②③ in green."""
    if not docs:
        print(f"  {_c(DIM)}(No matching experiences found){_c(RESET)}")
        return

    _CIRCLED = ["①", "②", "③", "④", "⑤"]
    try:
        print(f"\n{_c(BOLD)}📚 Retrieved Experiences (RAG){_c(RESET)}")
        print(f"{'─'*45}")
        for i, doc in enumerate(docs[:3]):
            num = _CIRCLED[i] if i < len(_CIRCLED) else f"{i+1}."
            # Truncate very long docs to keep terminal readable
            text = doc.strip().replace("\n", " ")
            if len(text) > 200:
                text = text[:197] + "..."
            print(f"  {_c(GREEN)}{num} {text}{_c(RESET)}")
    except UnicodeEncodeError:
        for i, doc in enumerate(docs[:3]):
            text = doc.strip().replace("\n", " ")[:200]
            print(f"  [{i+1}] {text}")


def print_plan(plan_text: str, label: str = "Combat Plan") -> None:
    """Print a plan block in cyan.

    label examples: "Combat Plan", "Continuing Plan from Combat Move"
    """
    if not plan_text or not plan_text.strip():
        return
    try:
        print(f"\n{_c(BOLD)}📋 {label}{_c(RESET)}")
        print(f"{'─'*45}")
        for line in plan_text.strip().split("\n"):
            print(f"  {_c(CYAN)}→ {line}{_c(RESET)}")
    except UnicodeEncodeError:
        print(f"\n[{label}]")
        print(plan_text.strip())


def print_action(action: str, reason: Optional[str] = None) -> None:
    """Print an action (orange) and optional one-line reason."""
    try:
        print(f"  {_c(ORANGE)}→ {action}{_c(RESET)}")
        if reason:
            print(f"  {_c(ORANGE)}  Why: {reason}{_c(RESET)}")
    except UnicodeEncodeError:
        print(f"  -> {action}")
        if reason:
            print(f"     Why: {reason}")


def print_action_section_header() -> None:
    """Print the '⚔️ Action' section header."""
    try:
        print(f"\n{_c(BOLD)}⚔️  Action{_c(RESET)}")
        print(f"{'─'*45}")
    except UnicodeEncodeError:
        print("\n[Action]")
        print("-" * 45)


def print_turn_end() -> None:
    """Print the end-turn confirmation line."""
    try:
        width = 44
        title = "PHASE 6 — END TURN"
        border = "═" * width
        padded = title.center(width)
        print(f"\n{_c(BOLD)}╔{border}╗{_c(RESET)}")
        print(f"{_c(BOLD)}║{padded}║{_c(RESET)}")
        print(f"{_c(BOLD)}╚{border}╝{_c(RESET)}")
        print(f"\n  {_c(GREEN)}✓ Turn ended.{_c(RESET)}")
    except UnicodeEncodeError:
        print("\n[END TURN]")
        print("  Turn ended.")


def print_transport_capacity(capacity_text: str) -> None:
    """Print the computed transport capacity summary."""
    if not capacity_text or not capacity_text.strip():
        return
    try:
        print(f"\n{_c(BOLD)}🚢 Transport Capacity (this round){_c(RESET)}")
        print(f"{'─'*45}")
        for line in capacity_text.strip().split("\n"):
            print(f"  {_c(CYAN)}{line}{_c(RESET)}")
    except UnicodeEncodeError:
        print("\n[Transport Capacity]")
        print(capacity_text.strip())


def print_phase_guard_warning(action_type: str, phase: str) -> None:
    """Print a phase-guard skip warning."""
    try:
        print(f"  {_c(RED)}[PHASE GUARD] '{action_type}' is not legal in {phase} — skipping.{_c(RESET)}")
    except UnicodeEncodeError:
        print(f"  [PHASE GUARD] '{action_type}' is not legal in {phase} — skipping.")


def print_deferred_landing(territory: str) -> None:
    """Print a deferred-landing notice."""
    try:
        print(f"  {_c(ORANGE)}[DEFERRED] Landing on {territory} deferred to Noncombat Move.{_c(RESET)}")
    except UnicodeEncodeError:
        print(f"  [DEFERRED] Landing on {territory} deferred to Noncombat Move.")
