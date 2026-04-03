#!/usr/bin/env python3
"""
Demo: drive Japanese via the bridge with a simple strategy.
Run the bridge first, then: python demo_bridge.py [--base-url http://localhost:8081] [--max-actions 20]
"""
from __future__ import annotations

import argparse
import json
import time

from triplea_bridge_client import TripleABridgeClient


def run_demo(base_url: str, max_actions: int) -> None:
    client = TripleABridgeClient(base_url=base_url)

    # Health check
    try:
        h = client.health()
        print(f"Bridge health: {h}")
    except Exception as e:
        print(f"Bridge not reachable at {base_url}: {e}")
        return

    state = client.get_state()
    game = state.get("game", {})
    # When not connected, bridge returns game.connected=false; when connected, we have currentPlayerName/territories
    if game.get("connected") is False and not state.get("territories"):
        print("Game not connected. Start host + bridge and take Japanese.")
        return

    our_player = game.get("currentPlayerName")
    japan = state.get("japan", {})
    pus = japan.get("pus", 0)
    print(f"Current player: {our_player}, Japanese PUs: {pus}")

    actions_done = 0
    while actions_done < max_actions:
        state = client.get_state()
        game = state.get("game", {})
        if game.get("connected") is False and not state.get("territories"):
            print("Game disconnected.")
            break
        current = game.get("currentPlayerName")
        step_name = game.get("stepName", "")

        legal = client.get_legal_actions()
        types = [a.get("type") for a in legal if a.get("type")]

        if not types:
            print("No legal actions (not our turn or waiting). Waiting 3s...")
            time.sleep(3)
            continue

        if "BUY_UNITS" in types:
            # Simple buy: spend some PUs on infantry
            japan = state.get("japan", {})
            pus = japan.get("pus", 0)
            purchase_options = state.get("purchaseOptions", [])
            if pus >= 3 and purchase_options:
                # Buy up to 3 infantry if we have option
                opt_names = [o.get("unitType") for o in purchase_options]
                if "infantry" in opt_names:
                    count = min(3, pus // 3)
                    if count > 0:
                        r = client.act_buy({"infantry": count})
                        if r.get("ok"):
                            print(f"  BUY_UNITS infantry x{count} -> {r}")
                            actions_done += 1
                        else:
                            print(f"  BUY_UNITS failed: {r.get('error')}")
            else:
                r = client.act_end_turn()
                if r.get("ok"):
                    print("  END_TURN (no purchase)")
                    actions_done += 1
                else:
                    print(f"  END_TURN failed: {r.get('error')}")
            time.sleep(0.5)
            continue

        if "PLACE_UNITS" in types:
            place_options = state.get("placeOptions", [])
            # placeOptions: list of { "territory": name, "maxPlaceCapacity": n or null }
            placements = []
            for opt in place_options:
                territory = opt.get("territory")
                if not territory:
                    continue
                # Try placing 1 infantry in this territory (we may have bought some)
                placements.append({"territory": territory, "unitType": "infantry", "count": 1})
                break
            if placements:
                r = client.act_place(placements)
                if r.get("ok"):
                    print(f"  PLACE_UNITS {placements} -> ok")
                    actions_done += 1
                else:
                    # No infantry to place or invalid; end turn
                    r = client.act_end_turn()
                    if r.get("ok"):
                        print("  END_TURN (place failed or done)")
                        actions_done += 1
            else:
                r = client.act_end_turn()
                if r.get("ok"):
                    print("  END_TURN (no place options)")
                    actions_done += 1
            time.sleep(0.5)
            continue

        # END_TURN only or other
        r = client.act_end_turn()
        if r.get("ok"):
            print(f"  END_TURN (step={step_name})")
            actions_done += 1
        else:
            print(f"  END_TURN failed: {r.get('error')}")
        time.sleep(0.5)

    print(f"Demo finished after {actions_done} actions.")


def main() -> None:
    p = argparse.ArgumentParser(description="TripleA bridge demo driver")
    p.add_argument("--base-url", default="http://localhost:8081", help="Bridge base URL")
    p.add_argument("--max-actions", type=int, default=20, help="Max actions to perform")
    args = p.parse_args()
    run_demo(args.base_url, args.max_actions)


if __name__ == "__main__":
    main()
