#!/usr/bin/env python3
"""
Chatgpt Driven combat, controlling 
"""
from __future__ import annotations

import argparse
import json
import os
import time

from triplea_bridge_client import TripleABridgeClient


JAPAN_PLAYER_NAMES = ("Japanese", "Japan")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_game_state",
            "description": "Get current TripleA game state: round, phase, current player, Japanese PUs, territories, units by territory, purchase options and place options.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_actions",
            "description": "Get list of legal action types for the current turn, e.g. BUY_UNITS, PLACE_UNITS, END_TURN. Only non-empty when it is Japanese's turn.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "do_action",
            "description": "Execute one action. BUY_UNITS: pass units. PLACE_UNITS: pass placements. PERFORM_MOVE: pass from, to, units (move units from one territory to another; use in combat or non-combat move phase). END_TURN: pass action_type only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["BUY_UNITS", "PLACE_UNITS", "PERFORM_MOVE", "END_TURN", "COMBAT_MOVE", "NON_COMBAT_MOVE", "BATTLE"],
                    },
                    "units": {
                        "type": "object",
                        "description": "Unit type to count for BUY_UNITS, e.g. {\"infantry\": 3}",
                    },
                    "placements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "territory": {"type": "string"},
                                "unitType": {"type": "string"},
                                "count": {"type": "integer"},
                            },
                            "required": ["territory", "unitType", "count"],
                        },
                    },
                    "from": {"type": "string", "description": "Territory name for PERFORM_MOVE (start). Must match state.territories[].name."},
                    "to": {"type": "string", "description": "Territory name for PERFORM_MOVE (destination). Must match state.territories[].name."},
                    "move_units": {
                        "type": "array",
                        "description": "For PERFORM_MOVE: list of {unitType, count} to move from 'from' to 'to'.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "unitType": {"type": "string"},
                                "count": {"type": "integer"},
                            },
                            "required": ["unitType", "count"],
                        },
                    },
                },
                "required": ["action_type"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是 TripleA 二战游戏中日本方的 AI。你只能通过下面三个工具与对局交互，不要猜测局面。

1. get_game_state：获取当前回合、阶段(stepName/phaseName)、日本 PUs、territories（含 name、owner、neighbors、unitsSummary）、unitsByTerritory、purchaseOptions、placeOptions 等。
2. get_legal_actions：获取当前可执行的动作类型（如 BUY_UNITS、PLACE_UNITS、PERFORM_MOVE、END_TURN）。
3. do_action：执行一个动作。action_type 必填。BUY_UNITS 填 units；PLACE_UNITS 填 placements；PERFORM_MOVE 填 from（起点领土）、to（终点领土）、move_units（[{unitType, count}]）；END_TURN 只填 action_type。

【移动与进攻】当 legal_actions 中有 PERFORM_MOVE（phase 为 combat 或 noncombat）时，你可以执行移动/进攻：用 do_action(action_type="PERFORM_MOVE", from="领土名", to="领土名", move_units=[{unitType, count}])。领土名必须与 state.territories[].name 完全一致；from 领土内必须有足够的己方单位（见 unitsSummary/unitsByTerritory）。战斗移动阶段可进攻敌占领土（to 为敌方领土即触发战斗）；非战斗移动可调兵、用运输船运兵等。可多次调用 PERFORM_MOVE 完成多批移动，最后调用 END_TURN 结束该阶段。

【重要】END_TURN 只结束「当前阶段」，不会结束整个日本回合。日本一回合包含多个阶段：购买 → 战斗移动 → 战斗 → 非战斗移动 → 部署新单位 → 收税等。在战斗移动/非战斗移动阶段：先根据局面执行若干次 PERFORM_MOVE（进攻、调兵、运输），再 END_TURN。

流程：每次用户说「继续」时，先 get_game_state 和 get_legal_actions。当 currentPlayerName === game.controlledPlayerName 时才是「我方回合」。若是我方回合：购买阶段做 BUY_UNITS 后 END_TURN；部署阶段做 PLACE_UNITS 后 END_TURN；战斗移动/非战斗移动阶段做若干 PERFORM_MOVE 后 END_TURN；其他阶段可 END_TURN。用简短中文说明每一步。"""


def run_tool(client: TripleABridgeClient, name: str, arguments: dict) -> dict:
    if name == "get_game_state":
        return client.get_state()
    if name == "get_legal_actions":
        return client.get_legal_actions()
    if name == "do_action":
        at = arguments.get("action_type")
        if at == "END_TURN" or at in ("COMBAT_MOVE", "NON_COMBAT_MOVE", "BATTLE"):
            return client.act_end_turn()
        if at == "BUY_UNITS":
            return client.act_buy(arguments.get("units") or {})
        if at == "PLACE_UNITS":
            return client.act_place(arguments.get("placements") or [])
        if at == "PERFORM_MOVE":
            return client.act_move(
                arguments.get("from") or "",
                arguments.get("to") or "",
                arguments.get("move_units") or arguments.get("units") or [],
            )
    return {"error": "unknown tool"}


def _call_chat_with_retry(openai_client, messages: list, max_retries: int = 5):
    """调用 chat.completions.create，遇 429 限速时等待后重试。"""
    import openai
    last_err = None
    for attempt in range(max_retries):
        try:
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except openai.RateLimitError as e:
            last_err = e
            wait = 5
            if e.response and hasattr(e.response, "headers"):
                retry_after = e.response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(2, int(retry_after))
            print(f"  [限速 429] 等待 {wait} 秒后重试 ({attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise last_err


def run_one_round(
    openai_client,
    client: TripleABridgeClient,
    messages: list,
    max_tool_rounds: int = 50,
) -> bool:
    """执行一轮模型对话（可能多次 tool 调用），直到模型不再发起 tool_calls。返回 True 表示本轮有执行过任何 tool。"""
    did_any = False
    for _ in range(max_tool_rounds):
        response = _call_chat_with_retry(openai_client, messages)
        msg = response.choices[0].message
        messages.append(msg)
        if msg.content:
            print("[GPT]", msg.content.strip() or "(无文字)")
        if not getattr(msg, "tool_calls", None):
            break
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = run_tool(client, name, arguments)
            did_any = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            if isinstance(result, dict):
                ok = result.get("ok")
                err = result.get("error")
                print(f"  [Tool] {name} -> ok={ok}" + (f" error={err}" if err else ""))
            else:
                print(f"  [Tool] {name} -> (返回 {type(result).__name__})")
    return did_any


def main() -> None:
    p = argparse.ArgumentParser(description="ChatGPT 驱动 TripleA 日本方")
    p.add_argument("--base-url", default="http://localhost:8081", help="Bridge 的 base URL")
    p.add_argument("--auto", action="store_true", help="全自动对战：循环「继续」直到 Ctrl+C")
    p.add_argument("--rules-file", default="", help="规则说明文件路径（文本/UTF-8），内容会作为游戏规则教给 GPT")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("请设置环境变量 OPENAI_API_KEY")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("请先安装: pip install openai")
        return

    client = TripleABridgeClient(base_url=args.base_url)
    try:
        client.health()
    except Exception as e:
        print(f"Bridge 不可达 ({args.base_url}): {e}")
        return
    print("Bridge 已连接，开始控制日方。")

    system_content = SYSTEM_PROMPT
    if args.rules_file and os.path.isfile(args.rules_file):
        try:
            with open(args.rules_file, "r", encoding="utf-8") as f:
                rules_text = f.read().strip()
            system_content = (
                "以下是你必须遵守的 TripleA 游戏规则（节选），请据此做购买与部署决策。\n\n"
                + rules_text
                + "\n\n---\n\n"
                + SYSTEM_PROMPT
            )
            print(f"已加载规则文件: {args.rules_file}")
        except Exception as e:
            print(f"读取规则文件失败 ({args.rules_file}): {e}")

    openai_client = OpenAI()
    messages: list = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": "请先调用 get_game_state 和 get_legal_actions，然后根据局面自动完成日本当前回合（购买、部署、结束回合）。用简短中文说明每一步。"
            if not args.auto
            else "请自动完成日本当前回合：先 get_game_state 和 get_legal_actions，再按需执行 BUY_UNITS / PLACE_UNITS / END_TURN，直到本回合结束。用简短中文说明。"
        },
    ]

    if not args.auto:
        run_one_round(openai_client, client, messages)
        print("单次执行结束")
        return

    # 全自动对战：在「用户消息」边界裁剪，避免 tool 消息脱离前面的 assistant+tool_calls 导致 400
    KEEP_LAST_USER_ROUNDS = 2  # 保留最近 2 个「用户回合」（每回合= user -> assistant -> tool* -> assistant）
    print("全自动对战已启动，按 Ctrl+C 停止。")
    try:
        while True:
            def _role(m):
                return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

            user_indices = [i for i, m in enumerate(messages) if _role(m) == "user"]
            if len(user_indices) > KEEP_LAST_USER_ROUNDS:
                start = user_indices[len(user_indices) - KEEP_LAST_USER_ROUNDS]
                messages = [messages[0]] + messages[start:]
                print("  [上下文已裁剪，保留最近若干回合]")
            run_one_round(openai_client, client, messages)
            try:
                state = client.get_state()
                game = state.get("game", {})
                cur = game.get("currentPlayerName")
                controlled = game.get("controlledPlayerName")
                # 优先用 Bridge 返回的 controlledPlayerName 判断是否我方回合；若无则回退到固定名单
                is_our_turn = (controlled is not None and cur == controlled) or (
                    controlled is None and cur in JAPAN_PLAYER_NAMES
                )
                legal_count = len(client.get_legal_actions())
                print(
                    f"  [Debug] currentPlayerName={cur!r} controlledPlayerName={controlled!r} "
                    f"是否我方回合={is_our_turn} legal_actions条数={legal_count}"
                )
                # 无地图且未连接：对局未开始或已断开，先等待再重试，不立即退出
                if not state.get("territories") and game.get("connected") is False:
                    wait_sec = 25
                    print(f"对局未开始或已断开，{wait_sec} 秒后再检查…")
                    time.sleep(wait_sec)
                # 若不是我方回合，等待较长时间再「继续」，避免疯狂轮询
                elif not is_our_turn:
                    wait_sec = 25
                    print(f"当前为 {cur} 回合（我方={controlled}），{wait_sec} 秒后再检查…")
                    time.sleep(wait_sec)
            except Exception:
                print("获取 state 失败，可能对局已断，退出。")
                break
            messages.append(
                {
                    "role": "user",
                    "content": "继续。若 currentPlayerName 等于 game.controlledPlayerName（即我方回合）则按阶段依次执行（购买→…→部署→各阶段 END_TURN），直到 currentPlayerName 变为他人；否则简短说明。",
                }
            )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
