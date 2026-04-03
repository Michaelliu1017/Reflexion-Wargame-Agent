# Reflexion-Wargame-Agent
A self-evolving wargame AI agent built on LLMs with RAG-based rule retrieval, Reflexion memory, and external tool integration. Achieves interpretable, training-free strategic reasoning in turn-based wargames.
## Quick Start Guide

> **Critical:** The three terminals must be started in order. Do not skip ahead.

---

## Prerequisites

- TripleA game bridge built and ready (`triplea-game-bridge/`)
- Python 3.11 virtual environment set up (`RRagent/agent/src/.venv/`)
- `.env` file configured at `RRagent/agent/.env`:
OPENAI_API_KEY=sk-xxxxxxx

- Map installed: `World War II Pacific 1940 2nd Edition`
  (If missing: open TripleA → **Maps → Download Maps** → search and download)

---

## Terminal 1 — TripleA Game (GUI)
```bash
 cd /Reflexion-Wargame-Agent/triplea-game-bridge         
./gradlew :game-app:game-headed:run
```

Once the GUI opens, follow these steps:

1. Click **Play → Start Local Game**
2. Select map: `World War II Pacific 1940 2nd Edition`
3. Configure players:
   - **Japan** → set to `Human` *(Bridge will take over this seat)*
   - **All other nations** → set to `Hard (AI)`
4. Click **Host** (top right) — the game will now wait for a client connection on port `3300`

> Do not proceed to Terminal 2 until the game is waiting for connections.

---

## Terminal 2 — Java Bridge
```bash
cd /Users/michaelliu/Documents/swe/LangChain_RAG_WargameBot/triplea-game-bridge
./gradlew :game-app:game-bridge:run --args="--host 127.0.0.1 --port 3300 --name Bot_Yamamoto --take Japanese"
```

You should see:
Bridge connected and took player: Japan
Bridge HTTP server listening on port 8081

The Japan seat in TripleA will be claimed and the game will start automatically.

> Do not proceed to Terminal 3 until you see both lines above.

---

## Terminal 3 — Python AI Agent
```bash
cd /Users/michaelliu/Documents/swe/LangChain_RAG_WargameBot/RRagent

# Activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install requirements
cd /LangChain_RAG_WargameBot/RRagent/agent
pip install -r requirements.txt

# Start agent
cd /LangChain_RAG_WargameBot/RRagent/agent/src
python main.py --milestone capture_india
```

You should see the ASCII banner followed by:
Waiting for Japanese turn...

The agent will now play automatically whenever it is Japan's turn.

---

## Available Milestones

| Milestone | Description | Max Rounds |
|---|---|---|
| `capture_india` | Capture Calcutta | 6 |
| `secure_sea_lanes` | Control Sumatra, Java, Borneo, Celebes | 5 |
| `destroy_us_navy` | Eliminate US fleet at Hawaiian Islands | 4 |
```bash
python main.py --milestone secure_sea_lanes
python main.py --milestone destroy_us_navy
```

---

## First-Time Setup

If `.venv` does not exist:
```bash
cd /Users/michaelliu/Documents/swe/LangChain_RAG_WargameBot/RRagent/agent
python3.11 -m venv src/.venv
source src/.venv/bin/activate
pip install -r requirements.txt
```

---

## Troubleshooting

**Bridge fails to connect**
Make sure Terminal 1 game is fully loaded and waiting on port 3300 before running Terminal 2.

**Agent not responding**
Confirm Terminal 2 shows `Bridge HTTP server listening on port 8081` before starting Terminal 3.

**Map not found**
Open TripleA → Maps → Download Maps → search `World War II Pacific 1940 2nd Edition`.

**ModuleNotFoundError**
Virtual environment not activated. Run `source ../.venv/bin/activate` before `python main.py`.

**API Key error**
Check that `RRagent/agent/.env` contains a valid `OPENAI_API_KEY`.
