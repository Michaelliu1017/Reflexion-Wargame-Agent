

# Reflexion-Wargame-Agent

A self-evolving wargame AI agent that learns from its own gameplay experience. Built on LLMs with a **hierarchical Reflexion** framework, **hybrid RAG** retrieval, and a **Criticizer** process supervisor — achieving interpretable, training-free strategic reasoning in TripleA's *World War II Pacific 1940* scenario.

<p align="center">
  <img src="assets/gitowl.png" width="100">
</p>

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Game Loop (main.py)                      │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Purchase │───▶│  Combat  │───▶│   NCM    │───▶│  Place   │   │
│  │          │    │  Move    │    │          │    │          │   │
│  └──────────┘    └────┬─────┘    └────┬─────┘    └──────────┘   │
│                       │               │                         │
│                  ┌────▼───────────────▼────┐                    │
│                  │   Criticizer Gate       │                    │
│                  │  (blocks premature end) │                    │
│                  └────────────────────────-┘                    │
│                                                                 │
│  Round Plan ◄── RAG retrieval ◄── Experience Pool               │
│       ▲                                 ▲                       │
│       │         ┌───────────────────────┤                       │
│       │         │                       │                       │
│  ┌────┴─────┐   │  ┌────────────────┐   │  ┌────────────────┐   │
│  │  Board   │   │  │  Round-level   │───┘  │  Game-level    │   │
│  │ Snapshot │───┘  │  Reflection    │      │  Reflection    │   │
│  └──────────┘      └────────────────┘      └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Dual-Layer Reflexion

| Layer | Trigger | What it does | Gate |
|-------|---------|-------------|------|
| **Round-level** | After each round (Config C) | Compares board snapshots; if score dropped or territory lost, LLM generates a tactical lesson → immediately indexed for same-game retrieval | Score improvement + new territory = skip |
| **Game-level** | After full game (Configs B, C) | Structured review of each strategic plan's execution → lessons persisted to `experiences.json` and `national_strategy.json` for cross-game learning | Always runs |

### Hybrid RAG (BM25 + FAISS + RRF)

Experience retrieval combines semantic search (FAISS) and keyword search (BM25) fused via Reciprocal Rank Fusion. Retrieved lessons are injected into:
- **Round Plan** — before each round (Configs B, C)
- **Phase instructions** — during Combat Move and NCM (Config C only)

### Criticizer Process Supervisor

An LLM-based gate embedded in `tool_end_turn()` that **blocks** premature phase advancement:

- **Combat Move**: Detects unattacked empty territories (free captures) and loaded transports that haven't performed amphibious assaults
- **NCM**: Detects idle transports and undeployed ground troops on Japan
- **Plan Generation**: Validates that new strategic plans target the China conquest objective

---

## Ablation Configurations

Three configurations for controlled experiments:

| Capability | A (Baseline) | B (RAG) | C (Full) |
|:-----------|:---:|:---:|:---:|
| Round Plan generation | ✓ | ✓ | ✓ |
| Game-level reflection + storage | — | ✓ | ✓ |
| RAG injection into Round Plan | — | ✓ | ✓ |
| Round-level reflection | — | — | ✓ |
| Per-phase RAG injection | — | — | ✓ |
| Criticizer gate | — | — | ✓ |

```bash
python main.py --config A   # Baseline
python main.py --config B   # RAG only
python main.py --config C   # Full system (default)
```

---

## Project Structure

```
LangChain_RAG_WargameBot/
├── RRagent/agent/
│   ├── src/
│   │   ├── main.py            # Game loop, config, milestone checks
│   │   ├── agent.py           # LLM agent, tools, Criticizer, Round Plan
│   │   ├── memory.py          # BoardSnapshot, Reflexion, GameMemory (RAG)
│   │   ├── bridge_client.py   # Python ↔ Java Bridge HTTP client
│   │   ├── battle_predictor.py# Monte Carlo battle odds simulation
│   │   └── display.py         # Terminal UI formatting
│   ├── knowledge/
│   │   ├── rules.txt          # Game rules (indexed by FAISS)
│   │   ├── national_strategy.json  # Evolving strategic plans
│   │   ├── rules_index/       # FAISS index for rules
│   │   └── exp_index/         # FAISS index for experiences
│   ├── memory/
│   │   └── experiences.json   # Persisted game lessons
│   ├── results/
│   │   └── game_results.csv   # Quantitative results per game
│   ├── requirements.txt
│   └── .env                   # OPENAI_API_KEY
│
├── triplea-game-bridge/
│   └── game-app/game-bridge/  # Java Bridge (BridgePlayer, BridgeRuntime)
│
└── assets/                    # Screenshots for documentation
```

---

## Quick Start

> The three terminals must be started in order. Do not skip ahead.

### Prerequisites

- Java 11+ (for TripleA + Bridge)
- Python 3.11+
- OpenAI API key
- TripleA map: `World War II Pacific 1940 2nd Edition`
  (If missing: open TripleA → **Maps → Download Maps** → search and install)

### Terminal 1 — TripleA Game (GUI)

```bash
cd triplea-game-bridge
./gradlew :game-app:game-headed:run
```

Once the GUI opens:
1. Click **Play → Start Local Game**
2. Select map: `World War II Pacific 1940 2nd Edition`
3. Set **Japan** → `Human`, all others → `Hard (AI)`
4. Click **Host** — game waits for connection on port `3300`

<img src="assets/step1.png" width="600" alt="TripleA GUI setup">

### Terminal 2 — Java Bridge

```bash
cd triplea-game-bridge
./gradlew :game-app:game-bridge:run \
  --args="--host 127.0.0.1 --port 3300 --name Bot_Yamamoto --take Japanese"
```

Wait for:
```
Bridge connected and took player: Japan
Bridge HTTP server listening on port 8081
```

<img src="assets/step2.png" width="600" alt="Bridge connected">

### Terminal 3 — Python AI Agent

```bash
cd RRagent/agent

# First time: create venv and install dependencies
python3.11 -m venv src/.venv
source src/.venv/bin/activate
pip install -r requirements.txt

# Configure API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run agent (default: Config C, full system)
cd src
python main.py --config C
```

<img src="assets/ingame.png" width="600" alt="Agent playing">

---

## CLI Reference

```bash
python main.py --config {A,B,C}   # Ablation configuration (default: C)
python main.py --clean             # Clear experience pool before start
```

| Flag | Description |
|------|-------------|
| `--config A` | Baseline — no reflection, no RAG |
| `--config B` | RAG — game-level reflection + experience retrieval |
| `--config C` | Full — round + game reflection, criticizer, per-phase RAG |
| `--clean` | Delete `exp_index/` and `experiences.json` (use when switching configs) |

---

## Scoring System

The agent's performance is evaluated quantitatively per round:

```
score = japan_controlled_china_ipc − china_army_remaining_value
```

- **japan_controlled_china_ipc**: Sum of IPC values of Chinese territories controlled by Japan
- **china_army_remaining_value**: Estimated IPC value of surviving Chinese military units

Higher score = better for Japan. Score trajectory is logged per round and saved to `results/game_results.csv`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bridge fails to connect | Ensure Terminal 1 game is fully loaded and waiting on port 3300 |
| Agent not responding | Confirm Terminal 2 shows `listening on port 8081` |
| Map not found | TripleA → Maps → Download → search `Pacific 1940 2nd Edition` |
| `ModuleNotFoundError` | Activate venv: `source src/.venv/bin/activate` |
| API key error | Check `RRagent/agent/.env` contains valid `OPENAI_API_KEY` |
| Stale experience data | Run with `--clean` to reset experience pool |

---

## License

See [LICENSE.txt](LICENSE.txt).
