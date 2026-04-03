"""
manage_memory.py — Interactive vector store management tool

Commands:
  view      List all documents in the vector store (rules + reflexion experience)
  search    Semantic keyword search
  list      List all game experiences in experiences.json
  delete    Delete a specific game_id from experiences.json and rebuild
  add       Manually add an experience to the vector store
  clear     Clear all reflexion experiences (keep rules), rebuild vector store
  rebuild   Full rebuild from rules.txt + experiences.json

Usage:
  python manage_memory.py view
  python manage_memory.py search "transport cycling"
  python manage_memory.py list
  python manage_memory.py delete <game_id>
  python manage_memory.py add "[Round 3][Purchase Units][Early][Southern Expansion] lesson text"
  python manage_memory.py clear
  python manage_memory.py rebuild
"""
import os
import sys
import json

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH    = os.path.join(PROJECT_ROOT, "knowledge", "rules.txt")
INDEX_PATH    = os.path.join(PROJECT_ROOT, "knowledge", "faiss_index")
EXP_PATH      = os.path.join(PROJECT_ROOT, "memory", "experiences.json")


def _load_memory():
    """Lazy import to avoid loading models for unrelated commands."""
    from memory import GameMemory
    return GameMemory(rules_path=RULES_PATH, index_path=INDEX_PATH)


def cmd_view(args):
    mem = _load_memory()
    docstore = mem.vectorstore.docstore._dict
    total = len(docstore)
    print(f"\nVector store contains {total} document chunks\n")

    rules_docs = []
    reflex_docs = []
    for uid, doc in docstore.items():
        src = doc.metadata.get("source", "rules")
        if src == "reflexion":
            reflex_docs.append((uid, doc))
        else:
            rules_docs.append((uid, doc))

    print(f"  📖 rules:     {len(rules_docs)} chunks")
    print(f"  🧠 reflexion: {len(reflex_docs)} chunks\n")

    show = "all"
    if args:
        show = args[0]  # "rules" / "reflexion" / "all"

    to_show = []
    if show in ("all", "rules"):
        to_show += rules_docs
    if show in ("all", "reflexion"):
        to_show += reflex_docs

    for i, (uid, doc) in enumerate(to_show, 1):
        src = doc.metadata.get("source", "rules")
        tag = "📖" if src == "rules" else "🧠"
        content = doc.page_content
        preview = content[:200].replace("\n", " ")
        if len(content) > 200:
            preview += "…"
        print(f"[{i:03d}] {tag} uid={uid[:8]}…")
        print(f"       {preview}")
        print()

    print(f"── Showing {len(to_show)} / {total} chunks ──")
    print("Tip: python manage_memory.py view rules      — show rules only")
    print("     python manage_memory.py view reflexion  — show experience only")


def cmd_search(args):
    if not args:
        print("Usage: python manage_memory.py search <keyword>")
        return
    query = " ".join(args)
    k = 5

    mem = _load_memory()
    docs = mem.vectorstore.similarity_search_with_score(query, k=k)

    print(f"\nSemantic search: '{query}'  Top {k} results\n")
    for rank, (doc, score) in enumerate(docs, 1):
        src = doc.metadata.get("source", "rules")
        tag = "📖" if src == "rules" else "🧠"
        sim = 1 - score  # approximate similarity from FAISS L2 distance
        print(f"[{rank}] {tag} similarity≈{sim:.3f}")
        print(f"    {doc.page_content[:300].replace(chr(10), ' ')}")
        print()


def cmd_list(args):
    if not os.path.exists(EXP_PATH):
        print(f"experiences.json not found: {EXP_PATH}")
        return

    with open(EXP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nexperiences.json contains {len(data)} game records\n")
    for i, game in enumerate(data, 1):
        gid     = game.get("game_id", "?")
        result  = game.get("result", "?")
        lessons = game.get("lessons", [])
        if lessons and isinstance(lessons[0], dict):
            n_in_rag = sum(1 for l in lessons if l.get("in_rag", False))
            n_total  = len(lessons)
            rag_info = f"in_rag: {n_in_rag}/{n_total}"
        else:
            rag_info = f"lessons: {len(lessons)}"
        print(f"[{i:02d}] game_id={gid}  {rag_info}")
        print(f"       result: {result[:80]}")
        for j, lesson in enumerate(lessons[:2], 1):
            txt = lesson if isinstance(lesson, str) else lesson.get("text", str(lesson))
            print(f"       lesson {j}: {txt[:120]}…")
        if len(lessons) > 2:
            print(f"       …({len(lessons)} total — use delete + rebuild to remove)")
        print()

    print("Tip: python manage_memory.py delete <game_id>  — delete game and rebuild")


def cmd_delete(args):
    if not args:
        print("Usage: python manage_memory.py delete <game_id>")
        return
    target_id = args[0]

    if not os.path.exists(EXP_PATH):
        print(f"experiences.json not found: {EXP_PATH}")
        return

    with open(EXP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    before = len(data)
    data = [g for g in data if g.get("game_id") != target_id]
    after = len(data)

    if before == after:
        print(f"game_id={target_id} not found — no changes made.")
        return

    with open(EXP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Deleted game_id={target_id}. {after} records remaining.")
    print("Rebuilding vector store...")
    cmd_rebuild([])


def cmd_add(args):
    if not args:
        print('Usage: python manage_memory.py add "[Round 3][Purchase Units][Early][Southern Expansion] lesson text"')
        return
    text = " ".join(args)
    mem = _load_memory()
    mem.add_experience(text)
    print(f"Added to vector store:\n  {text[:200]}")


def cmd_clear(args):
    confirm = input(
        "⚠ This will clear experiences.json and rebuild the vector store (rules kept).\n"
        "Type YES to confirm: > "
    ).strip()
    if confirm != "YES":
        print("Cancelled.")
        return

    with open(EXP_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)
    print("experiences.json cleared.")
    cmd_rebuild([])


def cmd_rebuild(args):
    import shutil
    from dotenv import load_dotenv
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()

    print("Building base vector store from rules.txt...")
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        rules_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", ".", ","],
    )
    docs = splitter.split_documents([Document(page_content=rules_text)])
    print(f"  Rules text: {len(docs)} chunks")

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(docs, embeddings)

    if os.path.exists(EXP_PATH):
        with open(EXP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        added = 0
        for game in data:
            lessons = game.get("lessons", [])
            for lesson in lessons:
                if isinstance(lesson, dict):
                    in_rag = lesson.get("in_rag", True)
                    text   = lesson.get("text", "")
                    score  = lesson.get("score", 5)
                    if not in_rag or not text or score < 5:
                        continue
                    meta = {
                        "source":         "reflexion",
                        "game_phase":     lesson.get("game_phase", ""),
                        "game_stage":     lesson.get("game_stage", ""),
                        "strategic_goal": lesson.get("strategic_goal", ""),
                        "round":          lesson.get("round", 0),
                    }
                else:
                    text = str(lesson)
                    meta = {"source": "reflexion"}

                if text.strip():
                    vs.add_documents([Document(page_content=text, metadata=meta)])
                    added += 1

        print(f"  Experiences: {added} entries added to RAG (in_rag=True and score≥5)")

    if os.path.exists(INDEX_PATH):
        backup = INDEX_PATH + "_backup"
        if os.path.exists(backup):
            shutil.rmtree(backup)
        shutil.copytree(INDEX_PATH, backup)
        print(f"  Old index backed up to {backup}")

    vs.save_local(INDEX_PATH)
    print(f"✓ Vector store rebuilt and saved to {INDEX_PATH}")


COMMANDS = {
    "view":    cmd_view,
    "search":  cmd_search,
    "list":    cmd_list,
    "delete":  cmd_delete,
    "add":     cmd_add,
    "clear":   cmd_clear,
    "rebuild": cmd_rebuild,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0)

    cmd  = sys.argv[1]
    rest = sys.argv[2:]
    COMMANDS[cmd](rest)
