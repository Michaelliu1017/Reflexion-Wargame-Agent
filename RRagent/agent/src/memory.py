"""
memory.py

Two classes:
  GameMemory      — RAG knowledge base for rules text and game experience, supports semantic retrieval
  ReflexionEngine — Post-game LLM reflection, extracts strategic lessons and stores them

RAG Retrieval Design:
  Only retrieves for 3 action phases: Purchase Units / Combat Move / Noncombat Move
  Query format matches stored text prefix exactly:
    "[Purchase Units][Early][Southern Expansion] round 3"
  Storage text format:
    "[Round 3][Purchase Units][Early][Southern Expansion] lesson..."

Reflection Pipeline (3-layer quality assurance):
  1. Structured generation — JSON Schema enforced format, 3 action-phase lessons per game
  2. Critic review — second LLM call with same API key, rewrites low-quality lessons
  3. Tiered storage — score ≥ 5 goes into FAISS (RAG retrievable) + JSON; all others JSON only
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

load_dotenv()


# ***************************************************************
# constant
# ***************************************************************

VALID_PHASES = ("Purchase Units", "Combat Move", "Noncombat Move")
VALID_STAGES = ("Early", "Mid", "Late")

_TERRITORY_HINTS = [
    "Japan", "Kiangsu", "Manchuria", "India", "Philippines", "Calcutta",
    "Burma", "Sumatra", "Java", "Borneo", "Celebes", "Sea Zone", "Hawaiian",
    "Malaya", "French Indo-China", "Kwangtung", "Hunan", "Kiangsi", "Anhwe",
    "Shantung", "Hupeh", "Yunnan", "Kwangsi", "Fukien", "Chekiang", "Hopei",
    "China", "Shensi",
]
_UNIT_HINTS = [
    "infantry", "transport", "armour", "artillery", "fighter",
    "bomber", "destroyer", "submarine", "carrier", "battleship",
    "tactical_bomber", "mech_infantry", "cruiser",
]


class ReflexionLesson(BaseModel):
    """
    A single game-phase-level strategic lesson.
    text field prefix format: [Round N][Phase][Stage][Strategic Goal]
    This ensures tag semantic weight is maximized during vector embedding for precise RAG retrieval.
    """
    game_phase: str = Field(
        description='Must be one of: "Purchase Units", "Combat Move", "Noncombat Move"'
    )
    game_stage: str = Field(
        description='"Early" (rounds 1-3), "Mid" (rounds 4-6), "Late" (rounds 7+)'
    )
    strategic_goal: str = Field(
        description='Overall campaign goal, e.g. "Southern Expansion", "Pacific Dominance"'
    )
    tactical_goal: str = Field(
        description='Specific tactical objective this lesson addresses, e.g. "Capture Kwangtung"'
    )
    round: int = Field(
        description="The game round this lesson primarily references"
    )
    text: str = Field(
        description=(
            "Full lesson text — this is the PRIMARY field used for RAG retrieval, so it must "
            "contain ALL key information. "
            "MUST start with: [Round N][Phase][Stage][Strategic Goal] "
            "Then include: (1) what happened, (2) specific unit counts, (3) what should have been done instead. "
            "BAD example: '[Round 2][Purchase Units][Early][Southern Expansion] Japan failed to purchase enough infantry.' "
            "(too vague, no numbers, no corrective action) "
            "GOOD example: '[Round 2][Purchase Units][Early][Southern Expansion] Japan bought 2 armour (12 PU) "
            "but only 1 infantry (3 PU), leaving 4 infantry on Japan with only 2 transports. "
            "Should have bought 1 transport (7 PU) + 2 infantry (6 PU) + 1 armour (6 PU) to resolve the "
            "transport bottleneck before stockpiling more land units.' "
            "The text MUST mention concrete numbers (unit counts, PU costs, territory names)."
        )
    )
    generalized_principle: str = Field(
        description=(
            "A reusable strategic principle abstracted from this specific lesson. "
            "Must be applicable across different games, not tied to exact territory states. "
            "BAD: 'Buy more infantry for Kwangtung' (too specific). "
            "GOOD: 'When japan_land ≥ 4 and transports ≤ 2, buy transports before infantry — "
            "island units have zero combat value until shipped.' "
            "GOOD: 'Always commit ≥ 1.5x defender count when attacking; if ratio < 1.5, "
            "add aircraft support or defer to next round.' "
            "GOOD: 'In NCM, dispatch ALL empty transports every round — an idle transport is 7 PU wasted.'"
        )
    )
    trigger: str = Field(
        description=(
            "Trigger condition: MUST include at least 2 specific territory names AND "
            "concrete unit counts for both sides. "
            'Example: "Round 3, Kwangtung has 2 UK infantry defenders, '
            'Japan has 4 infantry + 1 artillery in Kiangsi available"'
        )
    )
    mistake: str = Field(
        description=(
            "Specific mistake made this game: MUST include round number, territory name, "
            "and unit counts. "
            'Example: "Round 3 Purchase: bought 2 infantry instead of 3, '
            'leaving Kwangtung unreachable in round 4 Combat Move"'
        )
    )
    action: str = Field(
        description=(
            "Concrete in-game action to take next time. "
            "MUST reference real game actions (purchase / move / attack) with territory + unit count. "
            "NEVER write abstract actions like 'use diplomacy' or 'negotiate'. "
            'Example: "In Purchase Units phase of round N, buy 3 infantry when '
            'Kwangtung attack is planned for round N+1"'
        )
    )
    expected_outcome: str = Field(
        description=(
            "Quantifiable outcome: must include IPC gained, territory names, or force ratio. "
            'Example: "Capture Kwangtung (IPC +3), open route to French Indo-China next round"'
        )
    )
    legal_action_required: str = Field(
        description=(
            "Exact in-game action string. "
            'Examples: "purchase infantry x3", "attack Kwangtung with 4 infantry 1 artillery", '
            '"transport 2 infantry from Japan to 19 Sea Zone"'
        )
    )


class ReflexionOutput(BaseModel):
    """Complete structured reflection output covering all three action phases."""
    lessons: list[ReflexionLesson] = Field(
        description=(
            "3-6 reflection entries covering Purchase Units, Combat Move, and Noncombat Move. "
            "Each phase should have 1-2 entries based on actual game events."
        )
    )


class CriticVerdict(BaseModel):
    """Critic's verdict on a single lesson."""
    has_territory_names: bool = Field(
        description="Does 'trigger' contain at least 2 specific territory names?"
    )
    has_unit_counts: bool = Field(
        description="Does 'trigger' contain concrete unit counts for both sides?"
    )
    is_executable: bool = Field(
        description="Does 'action' describe a real in-game action with territory + unit count?"
    )
    has_specific_mistake: bool = Field(
        description="Does 'mistake' contain round number + territory + unit counts?"
    )
    text_has_numbers: bool = Field(
        description="Does 'text' contain at least 2 concrete numbers (unit counts or PU values)?"
    )
    principle_is_general: bool = Field(
        description="Is 'generalized_principle' applicable across different games (not tied to exact territories)?"
    )
    total_score: int = Field(description="Sum of the 6 booleans above (0-6)")
    needs_rewrite: bool = Field(description="True when total_score < 4")
    rewritten_trigger: Optional[str] = Field(
        default=None,
        description="Rewritten trigger with 2+ territories and unit counts"
    )
    rewritten_action: Optional[str] = Field(
        default=None,
        description="Rewritten action with territory name and unit count"
    )
    rewritten_mistake: Optional[str] = Field(
        default=None,
        description="Rewritten mistake with round number, territory, unit counts"
    )
    rewritten_legal_action: Optional[str] = Field(
        default=None,
        description="Rewritten legal_action_required as a specific game action string"
    )
    rewritten_text: Optional[str] = Field(
        default=None,
        description="Rewritten text with concrete numbers, unit counts, PU costs, and corrective action"
    )
    rewritten_principle: Optional[str] = Field(
        default=None,
        description="Rewritten generalized_principle that is reusable across games"
    )


# ***************************************************************
# GameMemory
# ***************************************************************

class GameMemory:
    # 1) initialize memory
    def __init__(
        self,
        rules_path: str = "./knowledge/rules.txt",
        rules_index_path: str = "./knowledge/rules_index",
        exp_index_path: str = "./knowledge/exp_index",
        experiences_json_path: Optional[str] = None,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.rules_index_path = rules_index_path
        self.exp_index_path = exp_index_path

        # ── Rules vector store (read-only after first build) ──
        if os.path.exists(rules_index_path):
            print(f"[Memory] Loading rules vector store: {rules_index_path}")
            self.rules_store = FAISS.load_local(
                rules_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print(f"[Memory] Building rules vector store from: {rules_path}")
            self.rules_store = self._build_from_rules(rules_path)
            self.rules_store.save_local(rules_index_path)
            print(f"[Memory] Rules vector store saved: {rules_index_path}")

        # ── Experience vector store + BM25 index ──
        self._bm25_corpus: list[str] = []
        self._bm25_index: BM25Okapi | None = None

        if os.path.exists(exp_index_path):
            print(f"[Memory] Loading experience vector store: {exp_index_path}")
            self.exp_store = FAISS.load_local(
                exp_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self._rebuild_bm25_from_faiss()
        else:
            self.exp_store = None
            if experiences_json_path and os.path.exists(experiences_json_path):
                self._seed_from_json(experiences_json_path)
            else:
                print(f"[Memory] No experience vector store (will create on first Reflexion)")

    # ── BM25 helpers ──

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase split + keep bracket tags like [Round 3] as single tokens."""
        tags = re.findall(r'\[[^\]]+\]', text)
        plain = re.sub(r'\[[^\]]+\]', '', text).lower().split()
        return [t.lower() for t in tags] + plain

    def _rebuild_bm25_from_faiss(self) -> None:
        """Rebuild BM25 index from all documents currently in the FAISS exp_store."""
        if self.exp_store is None:
            return
        all_docs = list(self.exp_store.docstore._dict.values())
        self._bm25_corpus = [doc.page_content for doc in all_docs]
        if self._bm25_corpus:
            tokenized = [self._tokenize(t) for t in self._bm25_corpus]
            self._bm25_index = BM25Okapi(tokenized)
            print(f"[Memory] BM25 index built: {len(self._bm25_corpus)} documents")
        else:
            self._bm25_index = None

    def _add_to_bm25(self, text: str) -> None:
        """Incrementally add a single document to the BM25 corpus and rebuild."""
        self._bm25_corpus.append(text)
        tokenized = [self._tokenize(t) for t in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)

    # ── Seed from JSON ──

    def _seed_from_json(self, json_path: str) -> None:
        """One-time import: load all in_rag lessons from experiences.json into exp_store + BM25."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read().strip() or "[]")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[Memory] Could not read {json_path} — skipping seed")
            return

        docs = []
        for game in data:
            for lesson in game.get("lessons", []):
                if not lesson.get("in_rag", False):
                    continue
                text = lesson.get("text", "")
                if not text.strip():
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": "reflexion",
                        "game_phase": lesson.get("game_phase", ""),
                        "game_stage": lesson.get("game_stage", ""),
                        "strategic_goal": lesson.get("strategic_goal", ""),
                        "round": lesson.get("round", 0),
                    },
                ))

        if docs:
            print(f"[Memory] Seeding exp vector store from {json_path}: {len(docs)} lessons (in_rag=true)")
            self.exp_store = FAISS.from_documents(docs, self.embeddings)
            self.exp_store.save_local(self.exp_index_path)
            print(f"[Memory] Experience vector store created and saved: {self.exp_index_path}")
            self._rebuild_bm25_from_faiss()
        else:
            print(f"[Memory] No in_rag lessons found in {json_path}")


    def _build_from_rules(self, rules_path: str) -> FAISS:
        with open(rules_path, "r", encoding="utf-8") as f:
            text = f.read()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,
            separators=["\n\n", "\n", ".", ","],
        )
        chunks = splitter.split_documents([Document(page_content=text)])
        print(f"[Memory] Rules text split into {len(chunks)} chunks, vectorizing...")
        return FAISS.from_documents(chunks, self.embeddings)

    # 2) retrieve memory（version 2） — Hybrid (BM25 + FAISS + RRF)
    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Hybrid retrieval (version 3):
          1. BM25 keyword search on exp_store  → top-k candidates
          2. FAISS semantic search on exp_store → top-k candidates
          3. Reciprocal Rank Fusion (RRF) merges both lists
          4. Rules store searched separately (semantic only, always appended)
        """
        exp_results: list[str] = []

        try:
            # ── Experience: Hybrid BM25 + FAISS ──
            if self.exp_store is not None:
                faiss_texts = self._faiss_exp_search(query, k=k)
                bm25_texts = self._bm25_search(query, k=k)
                exp_results = self._rrf_merge(faiss_texts, bm25_texts, k=k)

            # ── Rules: semantic only ──
            rules_docs = self.rules_store.similarity_search(query, k=2)
            rules_texts = [d.page_content for d in rules_docs]

        except Exception as e:
            err = str(e)
            if "insufficient_quota" in err:
                print(
                    "  [RAG] OpenAI quota exhausted — RAG retrieval skipped. "
                    "Please top up at platform.openai.com/account/billing."
                )
            elif "429" in err or "rate_limit" in err:
                print("  [RAG] Embedding API rate limited — skipping retrieval.")
            else:
                print(f"  [RAG] Retrieval failed ({type(e).__name__}) — skipping.")
            return ""

        combined = exp_results + rules_texts
        if not combined:
            return ""
        return "\n\n---\n\n".join(combined)

    def _faiss_exp_search(self, query: str, k: int) -> list[str]:
        if self.exp_store is None:
            return []
        docs = self.exp_store.similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def _bm25_search(self, query: str, k: int) -> list[str]:
        if self._bm25_index is None or not self._bm25_corpus:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._bm25_corpus[i] for i in top_indices if scores[i] > 0]

    @staticmethod
    def _rrf_merge(
        faiss_results: list[str],
        bm25_results: list[str],
        k: int = 3,
        rrf_k: int = 60,
    ) -> list[str]:
        """
        Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)) across both lists.
        rrf_k=60 is the standard constant from the original RRF paper.
        """
        scores: dict[str, float] = {}
        for rank, text in enumerate(faiss_results):
            scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, text in enumerate(bm25_results):
            scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank + 1)
        ranked = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        return ranked[:k]


    def add_experience(self, text: str, metadata: dict | None = None):
        """Add a high-quality experience entry to FAISS + BM25."""
        doc = Document(
            page_content=text,
            metadata={**(metadata or {}), "source": "reflexion"},
        )
        if self.exp_store is None:
            self.exp_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.exp_store.add_documents([doc])
        self.exp_store.save_local(self.exp_index_path)
        self._add_to_bm25(text)


# ***************************************************************
# ReflexionEngine (以下是reflexion相关代码，等待修改 4.14)
# ***************************************************************

class ReflexionEngine:

    def __init__(
        self,
        memory: GameMemory,
        experiences_path: str = "./memory/experiences.json",
        reflect_model: str = "gpt-4o",
        critic_model: str = "gpt-4o-mini",
    ):
        self.memory = memory
        self.experiences_path = experiences_path

        _base = ChatOpenAI(model=reflect_model, temperature=0.1)
        self.structured_llm = _base.with_structured_output(ReflexionOutput)

        _critic_base = ChatOpenAI(model=critic_model, temperature=0)
        self.critic_llm = _critic_base.with_structured_output(CriticVerdict)

        self.fallback_llm = ChatOpenAI(model=critic_model, temperature=0)

    # ── 429 限流重试 ──────────────────────────────────────────

    @staticmethod
    def _invoke_with_rate_limit_retry(fn, label: str = "LLM", max_retries: int = 5):
        import time
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                err = str(e)
                is_rl = (
                    "429" in err or "rate_limit_exceeded" in err
                    or "tokens per min" in err or "Rate limit" in err
                )
                if is_rl and attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    print(
                        f"  [Reflexion/{label}] Rate limit 429, waiting {wait}s before retry"
                        f" ({attempt + 1}/{max_retries - 1})..."
                    )
                    time.sleep(wait)
                    continue
                print(f"  [Reflexion/{label}] Call failed: {err[:120]}")
                return None
        return None

    # ── 主入口 ────────────────────────────────────────────────

    def reflect_and_store(
        self,
        game_id: str,
        game_log: list[str],
        result: str,
        round_num: int = 0,
        strategic_goal: str = "Southern Expansion",
    ) -> list[str]:
        """
        Call after game ends. Three-layer pipeline:
          1. Structured generation (JSON Schema enforced, 3 action-phase lessons)
          2. Critic review (rewrite low-quality lessons)
          3. Tiered storage (score ≥ 5 → FAISS + JSON; below 5 → JSON only)
        """
        game_stage = self._get_game_stage(round_num)
        print(
            f"\n[Reflexion] Starting reflection game={game_id} "
            f"round={round_num} stage={game_stage} goal={strategic_goal}"
        )
        print(f"[Reflexion] Result: {result}")

        prompt = self._build_prompt(game_log, result, round_num, game_stage, strategic_goal)

        raw = self._invoke_with_rate_limit_retry(
            lambda: self.structured_llm.invoke(prompt),
            label="structured-gen",
        )
        if raw is None:
            print("[Reflexion] ⚠ Structured generation failed — falling back to text mode...")
            return self._reflect_fallback(
                game_id, game_log, result, round_num, game_stage, strategic_goal
            )
        raw_lessons: list[ReflexionLesson] = raw.lessons
        print(f"[Reflexion] Generated {len(raw_lessons)} lessons")

        print("[Reflexion] Critic model reviewing...")
        refined: list[ReflexionLesson] = []
        rewrite_count = 0
        for lesson in raw_lessons:
            lesson, was_rewritten = self._critique(lesson)
            refined.append(lesson)
            if was_rewritten:
                rewrite_count += 1
        if rewrite_count:
            print(f"[Reflexion] Critic rewrote {rewrite_count} low-quality lessons")

        scored = [
            self._score(lesson, game_id, round_num)
            for lesson in refined
        ]
        self._print_score_report(scored)

        stored = 0
        for item in scored:
            if item["score"] >= 5:
                rag_text = item.get("rag_text", item["text"])
                self.memory.add_experience(
                    rag_text,
                    metadata={
                        "game_phase": item["game_phase"],
                        "game_stage": item["game_stage"],
                        "strategic_goal": item["strategic_goal"],
                        "round": item["round"],
                    },
                )
                stored += 1

        print(
            f"[Reflexion] {stored} lessons (score ≥ 5) added to RAG  |  "
            f"{len(scored) - stored} lessons (score < 5) JSON only"
        )
        self._save_to_json(game_id, result, round_num, strategic_goal, scored)
        return [s["text"] for s in scored]

    # ── 结构化 Prompt ─────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        game_log: list[str],
        result: str,
        round_num: int,
        game_stage: str,
        strategic_goal: str,
    ) -> str:
        log_text = "\n".join(game_log[-30:])   # 最近30条日志，控制 token
        n = 6  # 生成条数：3 个阶段 × 2 条

        return f"""You just completed rounds 1-{round_num} of Axis & Allies Pacific 1940 as Japan.
Game result: {result}
Current game stage: {game_stage} (rounds 1-3=Early, 4-6=Mid, 7+=Late)
Strategic Goal: {strategic_goal}

Game log (most recent actions):
{log_text}

Generate exactly {n} reflection entries. Each entry MUST be a valid JSON object.

Cover all 3 action phases (2 entries each):
  - "Purchase Units"   → what to buy to enable future attacks
  - "Combat Move"      → which territories to attack and with what forces
  - "Noncombat Move"   → how to move transports, reinforce, and reposition

Rules:
1. "game_phase" must be EXACTLY one of: "Purchase Units", "Combat Move", "Noncombat Move"
2. "game_stage" must be EXACTLY one of: "Early", "Mid", "Late"
3. "strategic_goal" must be: "{strategic_goal}"
4. "text" is the MOST IMPORTANT FIELD — it is used for RAG retrieval in future games.
   MUST start with: [Round N][phase][stage][{strategic_goal}]
   Then MUST include ALL of: (a) what happened, (b) specific unit counts and PU costs,
   (c) what should have been done instead with exact numbers.
   BAD: "[Round 2][Purchase Units][Early][Southern Expansion] Japan failed to buy enough infantry."
   GOOD: "[Round 2][Purchase Units][Early][Southern Expansion] Japan bought 2 armour (12 PU) + 1 infantry (3 PU) = 15 PU,
     leaving 5 infantry stuck on Japan with only 1 transport in 6 SZ. Should have bought 1 transport (7 PU) +
     1 armour (6 PU) + 1 infantry (3 PU) to fix the transport bottleneck."
5. "generalized_principle" must be a REUSABLE rule applicable in ANY game, not tied to specific territory states.
   BAD: "Buy more infantry for Kwangtung" (too specific, useless in a different game state)
   GOOD: "When japan_land >= 4 and transports <= 2, buy transports before infantry — island units = zero combat value"
   GOOD: "Always commit >= 1.5x defender count; if ratio < 1.5, add aircraft or defer to next round"
   GOOD: "Dispatch ALL empty transports every NCM — an idle transport is 7 PU wasted per round"
6. "action" and "legal_action_required" MUST reference real in-game actions:
   - purchase: "buy 3 infantry", "buy 1 transport"
   - attack: "attack Kwangtung with 3 infantry 1 artillery from Kiangsi"
   - move: "transport 2 infantry from Japan via 6 Sea Zone to 19 Sea Zone"
   NEVER write "use diplomacy", "negotiate", or any non-game action
7. "trigger" MUST contain: at least 2 territory names + unit counts for both sides
8. "mistake" MUST contain: round number + territory name + unit counts
9. "expected_outcome" MUST be quantifiable: IPC value, territory names, force ratio
10. Each entry must reference a DIFFERENT round or decision point from the log
11. "round" field is the game round the lesson primarily references (1 to {round_num})
"""

    #**************************************************************
    # Criticizer Model
    #**************************************************************
    def _critique(
        self,
        lesson: ReflexionLesson,
    ) -> tuple[ReflexionLesson, bool]:
        prompt = f"""You are a strict strategy coach reviewing a reflection entry for Axis & Allies Pacific 1940.

Entry to review:
  game_phase: {lesson.game_phase}
  text: {lesson.text}
  trigger: {lesson.trigger}
  action: {lesson.action}
  mistake: {lesson.mistake}
  legal_action_required: {lesson.legal_action_required}
  generalized_principle: {lesson.generalized_principle}

Evaluate these 6 criteria (True/False each):
1. has_territory_names: Does "trigger" contain at least 2 specific territory names?
2. has_unit_counts: Does "trigger" contain concrete unit counts for both sides?
3. is_executable: Does "action" describe a real game action with territory + unit type + count?
   (reject vague actions like "strengthen forces", "use transport efficiently")
4. has_specific_mistake: Does "mistake" contain round number + territory name + unit counts?
5. text_has_numbers: Does "text" contain at least 2 concrete numbers (unit counts, PU costs, or IPC values)?
   "Japan failed to purchase enough infantry" has ZERO numbers → False.
   "Japan bought 2 armour (12 PU) but had 5 infantry on Japan with only 1 transport" has 4 numbers → True.
6. principle_is_general: Is "generalized_principle" a reusable rule NOT tied to specific territories?
   "Buy more infantry for Kwangtung" → False (too specific).
   "When japan_land >= 4 and transports <= 2, prioritize transport purchases" → True (reusable).

If total_score < 4, provide rewritten versions for the failing fields.
Rewrite examples:
  trigger: "Round 4, Kwangtung defended by 2 UK infantry, Japan has 4 infantry + 1 artillery in Kiangsi"
  action: "In Combat Move round 4, move 4 infantry + 1 artillery from Kiangsi to attack Kwangtung"
  mistake: "Round 3 Purchase: bought 2 infantry instead of 3 for Kwangtung push, Kwangtung still UK-controlled in round 5"
  legal_action: "attack Kwangtung with 4 infantry 1 artillery"
  text: "[Round 3][Purchase Units][Early][Southern Expansion] Japan bought 2 infantry (6 PU) instead of 3 (9 PU), leaving only 2 infantry available to attack Kwangtung which has 3 UK defenders. Should have bought 3 infantry to achieve 1.5x force ratio."
  principle: "Always ensure attacking force >= 1.5x defenders before committing. If short, buy the deficit in Purchase or add aircraft support."
"""
        verdict = self._invoke_with_rate_limit_retry(
            lambda: self.critic_llm.invoke(prompt),
            label="critic",
        )
        if verdict is None or not verdict.needs_rewrite:
            return lesson, False

        rewritten = ReflexionLesson(
            game_phase=lesson.game_phase,
            game_stage=lesson.game_stage,
            strategic_goal=lesson.strategic_goal,
            tactical_goal=lesson.tactical_goal,
            round=lesson.round,
            text=verdict.rewritten_text or lesson.text,
            generalized_principle=verdict.rewritten_principle or lesson.generalized_principle,
            trigger=verdict.rewritten_trigger or lesson.trigger,
            mistake=verdict.rewritten_mistake or lesson.mistake,
            action=verdict.rewritten_action or lesson.action,
            expected_outcome=lesson.expected_outcome,
            legal_action_required=verdict.rewritten_legal_action or lesson.legal_action_required,
        )
        return rewritten, True

    # ── 评分 ──────────────────────────────────────────────────

    def _score(
        self,
        lesson: ReflexionLesson,
        game_id: str,
        round_num: int,
    ) -> dict:
        """
        Scoring dimensions (max 12 points, capped to 10 for star display):
          format          0-2  text prefix format correct + game_phase valid
          specificity     0-3  trigger contains number + territory name + unit name
          actionability   0-3  action contains territory + unit + legal_action non-empty
          reproducibility 0-2  mistake contains round number + territory name
          generalization  0-2  text has numbers + principle is general
        """
        text = lesson.text

        # format
        score_fmt = 0
        if re.search(r'\[Round \d+\]', text):                 score_fmt += 1
        if lesson.game_phase in VALID_PHASES:                  score_fmt += 1

        # specificity
        score_spc = 0
        trigger_all = lesson.trigger + " " + lesson.mistake
        if re.search(r'\d+', lesson.trigger):                   score_spc += 1
        if any(t in trigger_all for t in _TERRITORY_HINTS):    score_spc += 1
        if any(u in lesson.trigger for u in _UNIT_HINTS):      score_spc += 1

        # actionability
        score_act = 0
        if any(t in lesson.action for t in _TERRITORY_HINTS):  score_act += 1
        if any(u in lesson.action for u in _UNIT_HINTS):       score_act += 1
        if len(lesson.legal_action_required.strip()) > 10:     score_act += 1

        # reproducibility
        score_rep = 0
        mistake_lower = lesson.mistake.lower()
        if re.search(r'round \d+', mistake_lower) or re.search(r'\d+', lesson.mistake):
            score_rep += 1
        if any(t in lesson.mistake for t in _TERRITORY_HINTS): score_rep += 1

        # generalization (new dimension)
        score_gen = 0
        numbers_in_text = len(re.findall(r'\d+', text))
        if numbers_in_text >= 2:                               score_gen += 1
        principle = getattr(lesson, 'generalized_principle', '')
        if principle and len(principle) > 20 and not any(
            t in principle for t in ["Kwangtung", "Kiangsi", "Manchuria", "Borneo"]
        ):
            score_gen += 1

        total = min(10, score_fmt + score_spc + score_act + score_rep + score_gen)
        filled = min(5, round(total / 2))
        stars = "★" * filled + "☆" * (5 - filled)

        phase_short = {"Purchase Units": "purchase", "Combat Move": "combat",
                       "Noncombat Move": "ncm"}.get(lesson.game_phase, "misc")
        entry_id = f"exp_{game_id[:6]}_{round_num}_{phase_short}"

        rag_text = text
        if principle and principle.strip():
            rag_text = f"{text}\n[Principle] {principle.strip()}"

        return {
            "id":             entry_id,
            "game_phase":     lesson.game_phase,
            "game_stage":     lesson.game_stage,
            "strategic_goal": lesson.strategic_goal,
            "tactical_goal":  lesson.tactical_goal,
            "round":          lesson.round,
            "text":           text,
            "rag_text":       rag_text,
            "generalized_principle": principle,
            "structured": {
                "trigger":               lesson.trigger,
                "mistake":               lesson.mistake,
                "action":                lesson.action,
                "expected_outcome":      lesson.expected_outcome,
                "legal_action_required": lesson.legal_action_required,
            },
            "score":          total,
            "stars":          stars,
            "score_details": {
                "format":         score_fmt,
                "specificity":    score_spc,
                "actionability":  score_act,
                "reproducibility": score_rep,
                "generalization": score_gen,
            },
            "in_rag": total >= 5,
        }

    # ── 评分报告 ──────────────────────────────────────────────

    @staticmethod
    def _print_score_report(scored: list[dict]) -> None:
        if not scored:
            return
        avg = sum(s["score"] for s in scored) / len(scored)
        print("\n[Reflexion] Lesson quality score report")
        print(f"  {'#':<3} {'Score':<6} {'Stars':<10} {'Phase':<18} {'Preview'}")
        print(f"  {'─'*3} {'─'*6} {'─'*10} {'─'*18} {'─'*40}")
        for i, item in enumerate(scored, 1):
            preview = item["text"][:55].replace("\n", " ")
            if len(item["text"]) > 55:
                preview += "…"
            phase = item.get("game_phase", "?")[:16]
            print(f"  {i:<3} {item['score']:<6} {item['stars']:<10} {phase:<18} {preview}")
        print(f"  {'─'*3} {'─'*6} {'─'*10} {'─'*18} {'─'*40}")
        rag_n = sum(1 for s in scored if s["in_rag"])
        print(f"  Avg score: {avg:.1f}  |  ≥5 into RAG: {rag_n}  |  <5 JSON only: {len(scored)-rag_n}\n")

    # ── 辅助：回合阶段 ────────────────────────────────────────

    @staticmethod
    def _get_game_stage(round_num: int) -> str:
        if round_num <= 3:
            return "Early"
        elif round_num <= 6:
            return "Mid"
        else:
            return "Late"

    # ── 写入 JSON ─────────────────────────────────────────────

    def _save_to_json(
        self,
        game_id: str,
        result: str,
        round_num: int,
        strategic_goal: str,
        scored: list[dict],
    ) -> None:
        os.makedirs(os.path.dirname(self.experiences_path), exist_ok=True)

        existing = []
        if os.path.exists(self.experiences_path):
            with open(self.experiences_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)

        avg = sum(s["score"] for s in scored) / len(scored) if scored else 0
        existing.append({
            "game_id":        game_id,
            "result":         result,
            "round_num":      round_num,
            "game_stage":     self._get_game_stage(round_num),
            "strategic_goal": strategic_goal,
            "avg_score":      round(avg, 1),
            "rag_threshold":  5,
            "lessons":        scored,
        })

        with open(self.experiences_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        rag_n = sum(1 for s in scored if s["in_rag"])
        print(
            f"[Reflexion] Written to {self.experiences_path} "
            f"(avg score {avg:.1f}, {rag_n}/{len(scored)} in RAG)"
        )

    # ── 文本模式兜底 ──────────────────────────────────────────

    def _reflect_fallback(
        self,
        game_id: str,
        game_log: list[str],
        result: str,
        round_num: int,
        game_stage: str,
        strategic_goal: str,
    ) -> list[str]:
        log_text = "\n".join(game_log[-20:])
        prompt = (
            f"You completed round {round_num} of Axis & Allies Pacific 1940 as Japan.\n"
            f"Result: {result}\nStrategic Goal: {strategic_goal}\n\n"
            f"Game log:\n{log_text}\n\n"
            f"Write 3 lessons, one per action phase (Purchase Units / Combat Move / Noncombat Move).\n"
            f"Each lesson format (strictly follow):\n"
            f"[Round N][Phase][{game_stage}][{strategic_goal}] <lesson text with territory names and unit counts>"
        )
        response = self.fallback_llm.invoke(prompt)
        lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
        lessons = [l for l in lines if l.startswith("[Round")]
        if not lessons:
            lessons = lines[:3]

        # Simple scoring and storage
        scored = []
        for lesson in lessons:
            s_fmt = 1 if re.search(r'\[Round \d+\]', lesson) else 0
            s_spc = (1 if any(t in lesson for t in _TERRITORY_HINTS) else 0) + \
                    (1 if re.search(r'\d+', lesson) else 0)
            score = min(10, s_fmt + s_spc + 2)
            filled = min(5, round(score / 2))
            stars = "★" * filled + "☆" * (5 - filled)

            phase = "Combat Move"
            for p in VALID_PHASES:
                if p in lesson:
                    phase = p
                    break

            scored.append({
                "id":             f"exp_{game_id[:6]}_{round_num}_fallback",
                "game_phase":     phase,
                "game_stage":     game_stage,
                "strategic_goal": strategic_goal,
                "text":           lesson,
                "score":          score,
                "stars":          stars,
                "score_details":  {"format": s_fmt, "specificity": s_spc,
                                   "actionability": 2, "reproducibility": 0},
                "in_rag":         score >= 5,
                "structured":     None,
                "tactical_goal":  "",
                "round":          round_num,
            })

        self._print_score_report(scored)
        for item in scored:
            if item["in_rag"]:
                self.memory.add_experience(item["text"])

        self._save_to_json(game_id, result, round_num, strategic_goal, scored)
        return [s["text"] for s in scored]
