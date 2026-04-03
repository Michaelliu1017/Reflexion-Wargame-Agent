# Game Bridge (ChatGPT ↔ TripleA)

A **Java Bridge** that connects to a TripleA LAN host as a network client, takes one player (e.g. Japan), and exposes a local HTTP API for an LLM (e.g. ChatGPT) to read state and send actions.

## Prerequisites

- **Host**: Run TripleA GUI, host a LAN game (e.g. port **3300**), set Japan to **Human**, others to AI, then **Start**.
- **No password** on the host (bridge uses headless login without password).

## Run

```bash
./gradlew :game-app:game-bridge:run --args="--host 127.0.0.1 --port 3300 --name Bot_Michael --take Japan"
```

- **--host** / **--port**: Host address and port (must match the GUI host port).
- **--name**: Client player name (e.g. `Bot_Michael`).
- **--take**: Which seat to take (e.g. `Japan`).

Defaults: `--host 127.0.0.1`, `--port 3300`, `--name Bot_Bridge`, `--take Japan`.

HTTP server listens on **port 8080**.

---

## API (Phase 2)

### GET /health

Returns `ok`.

### GET /state

Observation for the LLM. JSON:

- **game**: `round`, `stepName`, `phaseName`, `currentPlayerName`
- **japan**: `pus`, `techTokens` (if any)
- **territories**: array of `{ name, owner, isLand, isWater, neighbors, unitsSummary, puValue? }`
- **unitsByTerritory**: map of territory name → `{ unitType: count }` for Japan’s units
- **purchaseOptions**: (only in Purchase phase) `[{ unitType, cost, maxAffordable }]`
- **placeOptions**: (only in Place phase) `[{ territory, maxPlaceCapacity }]`

### GET /legal_actions

Returns a list of valid action types for the current phase (only when it’s Japan’s turn):

- Not Japan’s turn → `[]`
- Purchase phase → `[{ "type": "BUY_UNITS" }, { "type": "END_TURN" }]`
- Place phase → `[{ "type": "PLACE_UNITS" }, { "type": "END_TURN" }]`
- Other phases → `[{ "type": "END_TURN" }]`

### POST /act

Apply one action. Body: JSON object with `type` and optional fields.

**END_TURN**

```json
{ "type": "END_TURN" }
```

**BUY_UNITS** (only in Purchase phase)

```json
{
  "type": "BUY_UNITS",
  "items": [
    { "unitType": "infantry", "count": 3 },
    { "unitType": "fighter", "count": 1 }
  ]
}
```

**PLACE_UNITS** (only in Place phase)

```json
{
  "type": "PLACE_UNITS",
  "placements": [
    { "territory": "Japan", "unitType": "infantry", "count": 3 },
    { "territory": "Manchuria", "unitType": "fighter", "count": 1 }
  ]
}
```

**Response**

- Success: `{ "ok": true }`
- Error: `{ "ok": false, "error": "..." }` (optional `details`)

---

## Example curl commands

```bash
# Health
curl -s localhost:8080/health

# Full state (observation)
curl -s localhost:8080/state | jq

# Legal actions
curl -s localhost:8080/legal_actions | jq

# End Japan’s turn
curl -s -X POST localhost:8080/act -H 'Content-Type: application/json' -d '{"type":"END_TURN"}'

# Buy units (during Purchase phase)
curl -s -X POST localhost:8080/act -H 'Content-Type: application/json' \
  -d '{"type":"BUY_UNITS","items":[{"unitType":"infantry","count":3}]}'

# Place units (during Place phase)
curl -s -X POST localhost:8080/act -H 'Content-Type: application/json' \
  -d '{"type":"PLACE_UNITS","placements":[{"territory":"Japan","unitType":"infantry","count":3}]}'
```

---

## Troubleshooting

- **Bridge won’t connect**: Ensure the GUI host is running and listening on the given port (e.g. 3300). No lobby required (LAN only).
- **"Not Japan's turn"**: Only Japan’s phases accept BUY_UNITS / PLACE_UNITS; use END_TURN when it’s Japan’s turn to end the step.
- **"Not in purchase phase" / "Not in place phase"**: Send BUY_UNITS only when `GET /state` shows a Purchase step for Japan; PLACE_UNITS only in a Place step.
- **"Unknown unit type"**: Use exact unit type names from the map (e.g. from `purchaseOptions` in `/state`). Case-insensitive.
- **"Not enough PUs"**: Check `japan.pus` in `/state` and `purchaseOptions[].cost` / `maxAffordable`.
- **Placement errors**: Ensure territory is in `placeOptions` and you have enough unplaced units of that type (from the previous purchase).

---

## Architecture

- **BridgeRuntime**: Connects via TripleA `ClientMessenger`, takes the chosen player, loads `GameData` on `doneSelectingPlayers`, and runs a `ClientGame` with **BridgePlayer** and **BridgeLaunchAction**. Exposes `getState()`, `getLegalActions()`, and `applyAction()`.
- **BridgePlayer**: Implements `Player`; blocks in `start(stepName)` until the bridge sends END_TURN (or the step is completed by BUY/PLACE). Uses `PlayerBridge.getRemoteDelegate()` to call the server’s Purchase/Place delegate for BUY_UNITS and PLACE_UNITS.
- **BridgeStateBuilder**: Builds the `/state` JSON (territories, unitsByTerritory, purchaseOptions, placeOptions) from `GameData` and optional place delegate.
- **BridgeLaunchAction**: Minimal no-UI `LaunchAction`; holds the game reference for state and lifecycle.
