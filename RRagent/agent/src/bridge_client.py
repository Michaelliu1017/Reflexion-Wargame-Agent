"""
bridge_client.py

HTTP client for communication with the TripleA Java Bridge.
The Bridge is a REST server embedded in the game; Python reads state and executes actions here.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────
BRIDGE_PORT = 8081
BRIDGE_URL = os.getenv("BRIDGE_URL", f"http://localhost:{BRIDGE_PORT}")


# ─────────────────────────────────────────────────────────────
# TripleABridgeClient
# 所有和游戏通信的逻辑都封装在这个类里
# ─────────────────────────────────────────────────────────────
class TripleABridgeClient:

    def __init__(self, base_url: str = BRIDGE_URL):
        self.base_url = base_url.rstrip("/")

    # ── 底层 HTTP 工具 ────────────────────────────────────────

    def _get(self, path: str, timeout: int = 30, max_retries: int = 0) -> Any:
        """发送 GET 请求，返回解析后的 JSON 对象。"""
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace") if e.fp else ""
                if body:
                    try:
                        out = json.loads(body)
                        if "error" in out:
                            print(f"[Bridge 错误] GET {path}: {e.code} - {out['error']}")
                    except Exception:
                        print(f"[Bridge 错误] GET {path}: {e.code} 响应体: {body[:200]}")
                raise
            except (TimeoutError, urllib.error.URLError) as e:
                last_err = e
                if attempt < max_retries:
                    continue
                raise last_err
        raise last_err

    def _post(self, path: str, body: dict[str, Any], timeout: int = 30, max_retries: int = 2) -> Any:
        """发送 POST 请求，body 为 Python dict，自动序列化为 JSON。"""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())
            except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
                last_err = e
                if attempt < max_retries:
                    continue
                if path == "/act":
                    return {"ok": False, "error": f"timeout after {max_retries + 1} attempts"}
                raise last_err

    # ── 基础接口 ──────────────────────────────────────────────

    def health(self) -> str:
        """GET /health 检查 Bridge 是否在运行，返回 'ok'。"""
        req = urllib.request.Request(f"{self.base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode().strip()

    def get_state(self) -> dict[str, Any]:
        """GET /state — fetch full game state (territories, units, IPC, phase, etc.)."""
        return self._get("/state", timeout=60, max_retries=2)

    def safe_get_state(self) -> dict[str, Any]:
        """
        Wrapper around get_state() that handles the non-fatal TripleA rendering error
        'Image Not Found: flags/Neutral_fade.gif'. Retries once after a 1-second delay.
        All other exceptions are re-raised.
        """
        try:
            return self.get_state()
        except Exception as e:
            _msg = str(e)
            if "Neutral_fade.gif" in _msg or "Image Not Found" in _msg:
                print(f"[WARNING] TripleA map rendering error (non-fatal): {e}")
                print("[WARNING] Retrying state fetch after 1 second...")
                time.sleep(1)
                return self.get_state()
            raise

    def get_legal_actions(self) -> list[dict[str, Any]]:
        """GET /legal_actions 获取当前阶段所有合法动作列表。"""
        return self._get("/legal_actions", timeout=45, max_retries=1)

    # ── 动作接口 ──────────────────────────────────────────────
    # *** 缺乏宣战act
    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """POST /act 执行任意动作，返回 {"ok": true} 或 {"ok": false, "error": "..."}。"""
        return self._post("/act", action, timeout=60, max_retries=2)

    def act_buy(self, units: dict[str, int]) -> dict[str, Any]:
        """
        购买阶段：购买单位。
        units 格式：{"infantry": 3, "fighter": 1}
        """
        items = [{"unitType": k, "count": v} for k, v in units.items() if v > 0]
        return self.act({"type": "BUY_UNITS", "items": items})

    def act_place(self, placements: list[dict[str, Any]]) -> dict[str, Any]:
        """
        部署阶段：把购买的单位放到领土上。
        placements 格式：[{"territory": "Japan", "unitType": "infantry", "count": 3}]
        """
        return self.act({"type": "PLACE_UNITS", "placements": placements})

    def act_move(self, from_territory: str, to_territory: str, units: list[dict[str, Any]]) -> dict[str, Any]:
        """
        移动/进攻：把单位从一个领土移动到另一个领土。
        units 格式：[{"unitType": "infantry", "count": 2}]
        战斗移动阶段移动到敌占领土会触发战斗。
        """
        return self.act({
            "type": "PERFORM_MOVE",
            "from": from_territory,
            "to": to_territory,
            "units": units,
        })

    def act_end_turn(self) -> dict[str, Any]:
        """结束当前阶段（不是整个回合，日本一回合有多个阶段）。"""
        return self.act({"type": "END_TURN"})

    # ── 阶段判断辅助 ──────────────────────────────────────────

    def get_phase(self) -> str:
        """
        返回当前 stepName（游戏内部阶段名）。
        例如：japanesePurchase / japaneseCombatMove / japaneseBattle 等。
        """
        state = self.get_state()
        return state.get("game", {}).get("stepName", "unknown")

    def is_our_turn(self) -> bool:
        """
        判断现在是否是日本方的回合。
        currentPlayerName == controlledPlayerName 时才是我方回合。
        整个日本回合结束后 currentPlayerName 会变成下一个玩家。
        Bridge 未连接时返回 False（不崩溃）。
        """
        try:
            state = self.get_state()
        except Exception:
            return False
        game = state.get("game", {})
        current = game.get("currentPlayerName", "")
        controlled = game.get("controlledPlayerName", "")
        if controlled:
            return current == controlled
        return current in ("Japanese", "Japan")


# ─────────────────────────────────────────────────────────────
# TripleA resource-file utility (Bug 4 fix)
# ─────────────────────────────────────────────────────────────

def ensure_neutral_fade_gif(triplea_root: str) -> None:
    """
    Creates a placeholder Neutral_fade.gif if it does not exist.
    TripleA throws an IllegalStateException when rendering neutral territories
    if this file is missing. Call this once at agent startup before the game begins.

    triplea_root: path to the TripleA game's asset directory
                  (the folder that contains the 'flags' subdirectory)
    """
    target = Path(triplea_root) / "flags" / "Neutral_fade.gif"
    if not target.exists():
        source = Path(triplea_root) / "flags" / "Neutral.gif"
        if source.exists():
            shutil.copy(source, target)
            print(f"[FIX] Created missing {target} by copying Neutral.gif")
        else:
            print(
                f"[WARNING] Cannot fix missing Neutral_fade.gif — "
                f"Neutral.gif also not found at {source}"
            )
    # else: file exists, nothing to do


# ─────────────────────────────────────────────────────────────
# Direct execution: quick connectivity test
# Usage: python bridge_client.py
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=== Bridge 连接测试 (port {BRIDGE_PORT}) ===\n")
    client = TripleABridgeClient()

    print("1. 检查连接 (health)...")
    try:
        h = client.health()
        print(f"   结果: {h}\n")
    except Exception as e:
        print(f"   失败: {e}")
        print("   请确认 TripleA 游戏和 Java Bridge 正在运行")
        exit(1)

    print("2. 获取游戏状态 (get_state)...")
    state = client.get_state()
    game = state.get("game", {})
    print(f"   当前玩家: {game.get('currentPlayerName')}")
    print(f"   当前阶段: {game.get('stepName')}")
    print(f"   日本 PUs: {state.get('japan', {}).get('pus')}\n")

    print("3. 获取合法动作 (get_legal_actions)...")
    actions = client.get_legal_actions()
    print(f"   可执行动作: {[a.get('type') for a in actions]}\n")

    print("=== 测试完成，Bridge 连接正常 ===")
