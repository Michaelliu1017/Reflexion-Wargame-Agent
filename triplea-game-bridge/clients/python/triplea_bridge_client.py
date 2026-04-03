"""
TripleA Bridge HTTP client. Requires bridge running (e.g. :game-app:game-bridge:run).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class TripleABridgeClient:
    """Client for the TripleA game-bridge HTTP API (GET /state, /legal_actions, POST /act)."""

    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip("/")

    def _get(
        self,
        path: str,
        timeout: int = 30,
        max_retries: int = 0,
    ) -> Any:
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
                            print(f"[Bridge 错误] GET {path}: {e.code} - {out['error']}", flush=True)
                    except Exception:
                        print(f"[Bridge 错误] GET {path}: {e.code} 响应体: {body[:500]}", flush=True)
                raise
            except (TimeoutError, urllib.error.URLError) as e:
                last_err = e
                if attempt < max_retries:
                    continue
                raise last_err
        raise last_err

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        timeout: int = 30,
        max_retries: int = 2,
    ) -> Any:
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
            except (TimeoutError, urllib.error.URLError) as e:
                last_err = e
                if attempt < max_retries:
                    continue
                if path == "/act":
                    return {"ok": False, "error": f"timeout after {max_retries + 1} attempts"}
                raise last_err

    def health(self) -> str:
        """GET /health. Returns 'ok' if bridge is up."""
        req = urllib.request.Request(f"{self.base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode().strip()

    def get_state(self) -> dict[str, Any]:
        """GET /state. Uses 60s timeout and 2 retries to tolerate turn/phase transitions."""
        return self._get("/state", timeout=60, max_retries=2)

    def get_legal_actions(self) -> list[dict[str, str]]:
        """GET /legal_actions. List of { \"type\": \"BUY_UNITS\" }, etc."""
        return self._get("/legal_actions", timeout=45, max_retries=1)

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """POST /act with given action. Returns { \"ok\": true } or { \"ok\": false, \"error\": \"...\" }.
        Uses 60s timeout and up to 2 retries on timeout to tolerate host delays (e.g. deploy phase)."""
        return self._post("/act", action, timeout=60, max_retries=2)

    def act_buy(self, units: dict[str, int]) -> dict[str, Any]:
        """Buy units. units e.g. {\"infantry\": 3, \"fighter\": 1}."""
        items = [{"unitType": k, "count": v} for k, v in units.items() if v > 0]
        return self.act({"type": "BUY_UNITS", "items": items})

    def act_place(self, placements: list[dict[str, Any]]) -> dict[str, Any]:
        """Place units. placements e.g. [{\"territory\": \"Japan\", \"unitType\": \"infantry\", \"count\": 3}]."""
        return self.act({"type": "PLACE_UNITS", "placements": placements})

    def act_end_turn(self) -> dict[str, Any]:
        """End current turn."""
        return self.act({"type": "END_TURN"})

    def act_move(
        self,
        from_territory: str,
        to_territory: str,
        units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform one move (combat or non-combat). units e.g. [{\"unitType\": \"infantry\", \"count\": 2}]."""
        return self.act({
            "type": "PERFORM_MOVE",
            "from": from_territory,
            "to": to_territory,
            "units": units,
        })


if __name__ == "__main__":
    client = TripleABridgeClient()
    print("health:", client.health())
    state = client.get_state()
    print("connected:", state.get("game", {}).get("connected"))
    print("currentPlayerName:", state.get("game", {}).get("currentPlayerName"))
    print("legal_actions:", client.get_legal_actions())
