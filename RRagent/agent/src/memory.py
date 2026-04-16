"""
memory.py

Three components:
  GameMemory        — RAG knowledge base (rules + experience), hybrid BM25 + FAISS retrieval
  NationalStrategy  — Persistent cross-game strategy document (load/save JSON)
  ReflexionEngine   — Post-game LLM reflection on Strategic Plans, updates NS

Strategic Plan Lifecycle:
  1. Game start  → _initialize_strategic_plans() creates plans from NS + board state
  2. Every round → Round Plan references active plans; Plan Tracker records progress
  3. Every 2 rds → _reassess_strategy() may add/modify/abandon plans
  4. Game end    → ReflexionEngine evaluates each plan → updates NS + stores experience

RAG Retrieval:
  Happens once at Round Plan generation (not per-phase).
  Query: strategic plan names + current game stage.
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
# Constants
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


# ***************************************************************
# National Strategy — persistent cross-game document
# ***************************************************************

def load_national_strategy(path: str) -> dict:
    """Load national_strategy.json; return empty structure if missing."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                print(f"[NS] Loaded national strategy v{data.get('version', '?')} "
                      f"({len(data.get('strategic_plans', []))} plans)")
                return data
    print("[NS] No national strategy found — will initialize from scratch")
    return {
        "nation": "Japan",
        "version": 0,
        "last_updated_game": None,
        "core_doctrine": "",
        "strategic_plans": [],
        "known_risks": [],
    }


def save_national_strategy(path: str, ns: dict) -> None:
    """Save national strategy back to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ns["version"] = ns.get("version", 0) + 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ns, f, ensure_ascii=False, indent=2)
    print(f"[NS] Saved national strategy v{ns['version']} → {path}")


def ns_to_prompt_text(ns: dict) -> str:
    """Format national strategy as text for LLM prompt injection."""
    if not ns or not ns.get("strategic_plans"):
        return ""
    lines = [f"=== NATIONAL STRATEGY ({ns.get('nation', 'Unknown')}) ==="]
    if ns.get("core_doctrine"):
        lines.append(f"Core Doctrine: {ns['core_doctrine']}")
    lines.append("")
    for sp in ns.get("strategic_plans", []):
        status = sp.get("status", "proposed")
        lines.append(f"[{status.upper()}] {sp['name']} (priority {sp.get('priority', '?')}, "
                      f"rounds {sp.get('target_rounds', '?')})")
        lines.append(f"  Reason: {sp.get('reason', 'N/A')}")
        for action in sp.get("key_actions", []):
            lines.append(f"  - {action}")
        lines.append(f"  Expected: {sp.get('expected_outcome', 'N/A')}")
        if sp.get("lessons_learned"):
            lines.append(f"  Lessons: {'; '.join(sp['lessons_learned'])}")
        lines.append("")
    if ns.get("known_risks"):
        lines.append("KNOWN RISKS:")
        for risk in ns["known_risks"]:
            lines.append(f"  - {risk.get('description', '')} → {risk.get('mitigation', '')}")
    return "\n".join(lines)


# ***************************************************************
# Strategic Plan data models (game-level, cross-round)
# ***************************************************************

class StrategicPlan(BaseModel):
    """A game-level strategic plan that persists across rounds."""
    plan_id: str = Field(description='Unique ID, e.g. "sp_secure_coast"')
    name: str = Field(description='Human-readable name, e.g. "Secure Chinese Coast"')
    reason: str = Field(description="Why this plan is important — strategic rationale")
    actions: list[str] = Field(description="Concrete actions needed to execute this plan")
    expected_outcome: str = Field(
        description="What success looks like, with territory names and timeline"
    )
    target_round: int = Field(
        description="Expected completion round (e.g. 3 means 'by end of round 3')"
    )
    status: str = Field(
        default="active",
        description='"active", "completed", "abandoned"',
    )
    progress: list[str] = Field(
        default_factory=list,
        description="Per-round progress entries, e.g. 'Round 2: captured Kiangsu'",
    )


class StrategicPlansInit(BaseModel):
    """LLM output when initializing strategic plans at game start."""
    plans: list[StrategicPlan] = Field(
        description="2-4 strategic plans based on national strategy and current board state"
    )


class StrategicPlanReview(BaseModel):
    """Post-game evaluation of a single Strategic Plan."""
    plan_id: str = Field(description="ID of the plan being reviewed")
    plan_name: str = Field(description="Name of the plan")
    achieved: bool = Field(description="Was the expected outcome achieved?")
    actual_outcome: str = Field(
        description="What actually happened — territory changes, IPC impact, etc."
    )
    failure_chain: str = Field(
        default="",
        description=(
            "If failed: causal chain explaining WHY. "
            'Example: "Failed to capture FIC by round 5 because Yunnan was never taken, '
            'blocking the land route. Chinese forces held Yunnan with 3 infantry while Japan '
            'had only 2 infantry in Kwangsi — insufficient force ratio."'
        ),
    )
    root_cause: str = Field(
        description=(
            '"strategy_error" — the plan itself was flawed (wrong target, wrong timing). '
            '"execution_error" — the plan was sound but a specific step failed.'
        )
    )
    root_cause_detail: str = Field(
        description="Which specific step or decision caused the failure"
    )
    lesson: str = Field(
        description=(
            "Reusable strategic lesson. Must be concrete and include territory/unit references. "
            'Example: "When pushing toward FIC, must secure Yunnan first (needs >= 3 inf vs '
            'Chinese defense). Skipping Yunnan leaves the southern corridor blocked."'
        )
    )
    national_strategy_update: str = Field(
        description=(
            "Recommended change to national_strategy.json. "
            'Examples: "Add known_risk: Yunnan blocks FIC push if not cleared", '
            '"Update sp_capture_fic: add Yunnan as prerequisite action", '
            '"Increase confidence for sp_secure_coast to 0.9"'
        )
    )


class StrategicReflexionOutput(BaseModel):
    """Complete post-game reflexion covering all strategic plans."""
    reviews: list[StrategicPlanReview] = Field(
        description="One review per strategic plan that was active during the game"
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
# ReflexionEngine — Strategic Plan-level post-game reflection
# ***************************************************************

class ReflexionEngine:

    def __init__(
        self,
        memory: GameMemory,
        experiences_path: str = "./memory/experiences.json",
        ns_path: str = "./knowledge/national_strategy.json",
        reflect_model: str = "gpt-4o",
    ):
        self.memory = memory
        self.experiences_path = experiences_path
        self.ns_path = ns_path

        _base = ChatOpenAI(model=reflect_model, temperature=0.1)
        self.structured_llm = _base.with_structured_output(StrategicReflexionOutput)
        self.fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── 429 rate-limit retry ─────────────────────────────────

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

    # ── Main entry point ─────────────────────────────────────

    def reflect_and_store(
        self,
        game_id: str,
        game_log: list[str],
        result: str,
        round_num: int,
        plan_tracker: list[StrategicPlan],
    ) -> list[StrategicPlanReview]:
        """
        Post-game strategic reflexion.
        Evaluates each Strategic Plan: achieved? failure chain? strategy vs execution error?
        Stores lessons in FAISS + JSON, updates national_strategy.json.
        """
        active_plans = [p for p in plan_tracker if p.status in ("active", "completed")]
        print(
            f"\n[Reflexion] Starting strategic reflection game={game_id} "
            f"round={round_num} plans={len(active_plans)}"
        )
        print(f"[Reflexion] Result: {result}")

        if not active_plans:
            print("[Reflexion] No active plans to reflect on — skipping")
            return []

        prompt = self._build_prompt(game_log, result, round_num, active_plans)

        raw = self._invoke_with_rate_limit_retry(
            lambda: self.structured_llm.invoke(prompt),
            label="strategic-reflexion",
        )

        if raw is None:
            print("[Reflexion] Structured generation failed — falling back to text mode")
            return self._reflect_fallback(game_id, game_log, result, round_num, active_plans)

        reviews: list[StrategicPlanReview] = raw.reviews
        print(f"[Reflexion] Generated {len(reviews)} plan reviews")

        self._print_review_report(reviews)

        # Store lessons in FAISS
        stored = 0
        for review in reviews:
            if review.lesson and review.lesson.strip():
                rag_text = (
                    f"[Plan: {review.plan_name}] "
                    f"[{'ACHIEVED' if review.achieved else 'FAILED'}] "
                    f"{review.lesson}"
                )
                self.memory.add_experience(
                    rag_text,
                    metadata={
                        "source": "strategic_reflexion",
                        "plan_id": review.plan_id,
                        "achieved": review.achieved,
                        "root_cause": review.root_cause,
                    },
                )
                stored += 1
        print(f"[Reflexion] {stored} strategic lessons added to RAG")

        # Save to experiences JSON
        self._save_to_json(game_id, result, round_num, plan_tracker, reviews)

        # Update national strategy
        self._update_national_strategy(game_id, reviews)

        return reviews

    # ── Prompt builder ───────────────────────────────────────

    @staticmethod
    def _build_prompt(
        game_log: list[str],
        result: str,
        round_num: int,
        plans: list[StrategicPlan],
    ) -> str:
        log_text = "\n".join(game_log[-40:])

        plans_text = ""
        for p in plans:
            progress_str = "\n    ".join(p.progress) if p.progress else "(no progress recorded)"
            plans_text += f"""
  Plan: {p.name} (id: {p.plan_id})
    Status: {p.status}
    Reason: {p.reason}
    Actions planned: {'; '.join(p.actions)}
    Expected outcome: {p.expected_outcome}
    Target round: {p.target_round}
    Progress log:
    {progress_str}
"""

        return f"""You just completed rounds 1-{round_num} of Axis & Allies Pacific 1940.
Game result: {result}

The following Strategic Plans were active during this game:
{plans_text}

Game log (most recent actions):
{log_text}

For EACH strategic plan above, generate a review. Your analysis chain for each plan:

1. OUTCOME CHECK: Was the expected outcome achieved? Check territories, IPC, unit positions.

2. If FAILED — FAILURE CHAIN ANALYSIS:
   Trace the causal chain backward. Example:
   "Failed to capture FIC by round 5 ← Yunnan was never taken ← only 2 infantry in Kwangsi vs 3 Chinese defenders ← insufficient force allocation to southern front"
   Be specific: territory names, unit counts, round numbers.

3. ROOT CAUSE CLASSIFICATION:
   "strategy_error" — the plan itself was flawed (wrong target, unrealistic timeline, wrong prerequisite order)
   "execution_error" — the plan was sound but a specific tactical step failed (bad force ratio, forgot to move units, wrong purchase)

4. ROOT CAUSE DETAIL: Which exact step or decision was the root cause?

5. LESSON: A reusable strategic insight for future games. Must be concrete.
   BAD: "Should have planned better"
   GOOD: "When pushing toward FIC, Yunnan must be cleared first (requires >= 3 infantry). Add 'Secure Yunnan' as prerequisite action before FIC assault."

6. NATIONAL STRATEGY UPDATE: What should change in the persistent strategy document?
   Examples: "Add Yunnan as prerequisite for FIC push", "Increase confidence for coastal plan to 0.9", "Add risk: UK reinforces India if FIC not taken by round 5"
"""

    # ── Review report ────────────────────────────────────────

    @staticmethod
    def _print_review_report(reviews: list[StrategicPlanReview]) -> None:
        if not reviews:
            return
        print("\n[Reflexion] Strategic Plan Review Report")
        print(f"  {'#':<3} {'Result':<10} {'Root Cause':<18} {'Plan'}")
        print(f"  {'─'*3} {'─'*10} {'─'*18} {'─'*40}")
        for i, r in enumerate(reviews, 1):
            status = "ACHIEVED" if r.achieved else "FAILED"
            print(f"  {i:<3} {status:<10} {r.root_cause:<18} {r.plan_name}")
            if not r.achieved and r.failure_chain:
                chain_preview = r.failure_chain[:80]
                if len(r.failure_chain) > 80:
                    chain_preview += "..."
                print(f"      Chain: {chain_preview}")
            if r.lesson:
                lesson_preview = r.lesson[:80]
                if len(r.lesson) > 80:
                    lesson_preview += "..."
                print(f"      Lesson: {lesson_preview}")
        achieved_n = sum(1 for r in reviews if r.achieved)
        print(f"  {'─'*3} {'─'*10} {'─'*18} {'─'*40}")
        print(f"  Achieved: {achieved_n}/{len(reviews)}\n")

    # ── Save to experiences JSON ─────────────────────────────

    def _save_to_json(
        self,
        game_id: str,
        result: str,
        round_num: int,
        plan_tracker: list[StrategicPlan],
        reviews: list[StrategicPlanReview],
    ) -> None:
        os.makedirs(os.path.dirname(self.experiences_path), exist_ok=True)

        existing = []
        if os.path.exists(self.experiences_path):
            with open(self.experiences_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)

        plans_data = [p.model_dump() for p in plan_tracker]
        reviews_data = [r.model_dump() for r in reviews]

        existing.append({
            "game_id": game_id,
            "result": result,
            "round_num": round_num,
            "strategic_plans": plans_data,
            "reviews": reviews_data,
        })

        with open(self.experiences_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        print(f"[Reflexion] Written to {self.experiences_path}")

    # ── Update national strategy ─────────────────────────────

    def _update_national_strategy(
        self,
        game_id: str,
        reviews: list[StrategicPlanReview],
    ) -> None:
        ns = load_national_strategy(self.ns_path)
        ns["last_updated_game"] = game_id

        ns_plans = {sp["id"]: sp for sp in ns.get("strategic_plans", [])}

        for review in reviews:
            sp = ns_plans.get(review.plan_id)
            if sp is None:
                continue

            if review.achieved:
                sp["status"] = "validated"
                sp["confidence"] = min(1.0, sp.get("confidence", 0.5) + 0.15)
            else:
                sp["confidence"] = max(0.1, sp.get("confidence", 0.5) - 0.1)

            if review.lesson and review.lesson.strip():
                lessons = sp.get("lessons_learned", [])
                lessons.append(review.lesson.strip())
                sp["lessons_learned"] = lessons[-5:]

            if review.national_strategy_update and review.national_strategy_update.strip():
                update_text = review.national_strategy_update.strip()
                if update_text.lower().startswith("add risk"):
                    risks = ns.get("known_risks", [])
                    risks.append({
                        "description": update_text,
                        "mitigation": "(auto-generated from reflexion)",
                    })
                    ns["known_risks"] = risks

        ns["strategic_plans"] = list(ns_plans.values())
        save_national_strategy(self.ns_path, ns)

    # ── Fallback: text-mode reflexion ────────────────────────

    def _reflect_fallback(
        self,
        game_id: str,
        game_log: list[str],
        result: str,
        round_num: int,
        plans: list[StrategicPlan],
    ) -> list[StrategicPlanReview]:
        plan_names = ", ".join(p.name for p in plans)
        log_text = "\n".join(game_log[-20:])
        prompt = (
            f"You completed round {round_num} of Axis & Allies Pacific 1940.\n"
            f"Result: {result}\n"
            f"Active plans: {plan_names}\n\n"
            f"Game log:\n{log_text}\n\n"
            f"For each plan, write one line: [Plan Name] ACHIEVED/FAILED — reason — lesson"
        )
        response = self.fallback_llm.invoke(prompt)
        lines = [ln.strip() for ln in response.content.strip().split("\n") if ln.strip()]

        reviews = []
        for i, plan in enumerate(plans):
            line = lines[i] if i < len(lines) else f"{plan.name} — FAILED — no data"
            achieved = "ACHIEVED" in line.upper()
            reviews.append(StrategicPlanReview(
                plan_id=plan.plan_id,
                plan_name=plan.name,
                achieved=achieved,
                actual_outcome=line,
                failure_chain="" if achieved else line,
                root_cause="execution_error",
                root_cause_detail="(fallback mode — detail unavailable)",
                lesson=line,
                national_strategy_update="",
            ))

        self._print_review_report(reviews)

        for review in reviews:
            if review.lesson:
                self.memory.add_experience(
                    f"[Plan: {review.plan_name}] {review.lesson}",
                    metadata={"source": "strategic_reflexion", "plan_id": review.plan_id},
                )

        self._save_to_json(game_id, result, round_num, plans, reviews)
        self._update_national_strategy(game_id, reviews)
        return reviews
