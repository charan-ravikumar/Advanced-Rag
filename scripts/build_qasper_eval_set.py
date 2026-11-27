"""
Qasper Evaluation Set Builder
=============================
Builds a reproducible ~200-paper corpus and a golden question set from the
official Qasper v0.3 DEV split (https://allenai.org/data/qasper, CC BY 4.0).

Method:
    1. Select questions first (answerable + unanswerable) with a fixed seed.
    2. Include every paper referenced by those questions' gold evidence.
    3. Fill the remainder up to N_PAPERS with random distractor papers.

Outputs (eval/datasets/qasper_dev/):
    corpus.json     — Qasper-format papers; ingest with QASPER_CORPUS_PATH
    golden_set.json — questions, reference answers, gold evidence
    README.md       — source, license, method and counts

Usage (run from project root):
    python -m scripts.build_qasper_eval_set
    python -m scripts.build_qasper_eval_set --papers 200 --answerable 80 --unanswerable 20
"""

# ============================================
# IMPORTS
# ============================================

import argparse
import json
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

# ============================================
# CONFIG
# ============================================

ROOT = Path(__file__).resolve().parent.parent

QASPER_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com"
    "/qasper-train-dev-v0.3.tgz"
)

CACHE_DIR = ROOT / ".cache" / "qasper"
DEV_FILE = CACHE_DIR / "qasper-dev-v0.3.json"
OUTPUT_DIR = ROOT / "eval" / "datasets" / "qasper_dev"

SEED = 42
N_PAPERS = 200
N_ANSWERABLE = 80
N_UNANSWERABLE = 20
MAX_Q_PER_PAPER = 2

UNANSWERABLE_REFERENCE = "I could not find that information in the provided documents."


# ============================================
# DOWNLOAD
# ============================================

def download_dev_split() -> None:
    """Download and extract the Qasper archive into CACHE_DIR if not cached."""
    if DEV_FILE.is_file():
        print(f"Using cached {DEV_FILE}")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / "qasper-train-dev-v0.3.tgz"

    print(f"Downloading {QASPER_URL}")
    with urllib.request.urlopen(QASPER_URL, timeout=120) as resp, open(archive, "wb") as out:
        shutil.copyfileobj(resp, out)

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=CACHE_DIR, filter="data")

    archive.unlink()
    print(f"Extracted to {CACHE_DIR}")


# ============================================
# QUESTION PARSING
# ============================================

def _answer_text(answer: dict) -> str | None:
    """Return the reference answer string for one Qasper annotation, or None."""
    if answer.get("extractive_spans"):
        return " ".join(answer["extractive_spans"])
    if answer.get("free_form_answer"):
        return answer["free_form_answer"]
    if answer.get("yes_no") is not None:
        return "yes" if answer["yes_no"] else "no"
    return None


def _paper_paragraphs(paper: dict) -> set[str]:
    """All retrievable paragraphs of a paper (abstract + body)."""
    paras = {paper.get("abstract", "").strip()}
    for section in paper.get("full_text", []):
        paras.update(p.strip() for p in section.get("paragraphs", []))
    return paras


def parse_question(paper_id: str, paper: dict, qa: dict) -> dict | None:
    """
    Convert one Qasper QA into a golden-set record, or None if it is unusable.

    Questions whose annotators disagree on answerability are dropped. Evidence
    that is a table/figure caption (FLOAT SELECTED) is dropped because
    paper_to_document only ingests abstract + body text.
    """
    question = (qa.get("question") or "").strip()
    annotations = [a.get("answer", {}) for a in qa.get("answers", [])]
    if not question or not annotations:
        return None

    base = {
        "id": f"{paper_id}:{qa.get('question_id', '')}",
        "paper_id": paper_id,
        "paper_title": paper.get("title", ""),
        "question": question,
    }

    unanswerable = [a.get("unanswerable", False) for a in annotations]
    if all(unanswerable):
        return {
            **base,
            "answerable": False,
            "gold_answer": UNANSWERABLE_REFERENCE,
            "gold_answers": [UNANSWERABLE_REFERENCE],
            "gold_evidence": [],
        }
    if any(unanswerable):
        return None

    paragraphs = _paper_paragraphs(paper)
    answers, evidence = [], []
    for a in annotations:
        text = _answer_text(a)
        if text:
            answers.append(text)
        for ev in a.get("evidence") or []:
            ev = ev.strip()
            if ev and not ev.startswith("FLOAT SELECTED") and ev in paragraphs and ev not in evidence:
                evidence.append(ev)

    if not answers or not evidence:
        return None

    return {
        **base,
        "answerable": True,
        "gold_answer": answers[0],
        "gold_answers": list(dict.fromkeys(answers)),
        "gold_evidence": evidence,
    }


# ============================================
# SUBSET SELECTION
# ============================================

def select_questions(raw: dict, rng: random.Random, n_answerable: int, n_unanswerable: int) -> list[dict]:
    """Sample questions (capped per paper) before any document selection."""
    answerable_pool, unanswerable_pool = [], []
    for paper_id in sorted(raw):
        for qa in raw[paper_id].get("qas", []):
            record = parse_question(paper_id, raw[paper_id], qa)
            if record is None:
                continue
            (answerable_pool if record["answerable"] else unanswerable_pool).append(record)

    selected, per_paper = [], {}
    for pool, n in ((answerable_pool, n_answerable), (unanswerable_pool, n_unanswerable)):
        rng.shuffle(pool)
        taken = 0
        for record in pool:
            if taken == n:
                break
            if per_paper.get(record["paper_id"], 0) >= MAX_Q_PER_PAPER:
                continue
            per_paper[record["paper_id"]] = per_paper.get(record["paper_id"], 0) + 1
            selected.append(record)
            taken += 1
        if taken < n:
            raise RuntimeError(f"Only {taken} questions available, wanted {n}")

    return selected


def build_corpus(raw: dict, questions: list[dict], rng: random.Random, n_papers: int) -> dict:
    """Gold papers first, then seeded random distractors up to n_papers."""
    gold_ids = sorted({q["paper_id"] for q in questions})
    if len(gold_ids) > n_papers:
        raise RuntimeError(
            f"{len(gold_ids)} gold papers exceed the {n_papers}-paper budget; "
            "reduce the question counts."
        )

    remaining = sorted(set(raw) - set(gold_ids))
    distractor_ids = rng.sample(remaining, min(n_papers - len(gold_ids), len(remaining)))

    corpus = {}
    for paper_id in gold_ids + sorted(distractor_ids):
        paper = raw[paper_id]
        corpus[paper_id] = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "full_text": paper.get("full_text", []),
            "qas": [],
            "is_gold": paper_id in gold_ids,
        }
    return corpus


# ============================================
# OUTPUT
# ============================================

def write_readme(path: Path, meta: dict) -> None:
    path.write_text(f"""# Qasper dev evaluation set

**Source:** Qasper v0.3, dev split — Dasigi et al., 2021, "A Dataset of
Information-Seeking Questions and Answers Anchored in Research Papers"
(<https://allenai.org/data/qasper>).
**License:** CC BY 4.0.

## Files

| File | Contents |
|------|----------|
| `corpus.json` | {meta['n_documents']} papers in Qasper format (`title`, `abstract`, `full_text`). Questions are stripped. `is_gold` marks papers referenced by gold evidence. |
| `golden_set.json` | {meta['n_questions']} questions with reference answers and gold evidence. |

## Counts

| | |
|---|---|
| Documents | {meta['n_documents']} ({meta['n_gold_documents']} gold + {meta['n_distractors']} distractors) |
| Questions | {meta['n_questions']} ({meta['n_answerable']} answerable, {meta['n_unanswerable']} unanswerable) |

## Subsetting method

Generated by `python -m scripts.build_qasper_eval_set` with seed {meta['seed']}:

1. Questions are sampled first, at most {MAX_Q_PER_PAPER} per paper. Questions where annotators
   disagree on answerability are excluded.
2. Every paper referenced by a selected question is included.
3. Remaining slots are filled with randomly sampled dev papers as distractors.

Answerable questions require at least one body/abstract evidence paragraph that appears verbatim in the paper.
Table/figure-caption evidence (`FLOAT SELECTED`) is dropped because the ingestion path does not index it.
Unanswerable questions have empty `gold_evidence` and the pipeline's refusal sentence as the reference answer.

## Golden set schema

`id`, `paper_id`, `paper_title`, `question`, `answerable`, `gold_answer`, `gold_answers`
(all annotators), `gold_evidence` (list of passage texts).

## Usage

```bash
python -m scripts.build_qasper_eval_set               # rebuild (deterministic)
QASPER_CORPUS_PATH=eval/datasets/qasper_dev/corpus.json python -m scripts.ingest_qasper
python -m eval.run_eval                               # retrieval metrics
python -m eval.run_eval --with-answers                # + answer quality
```

`scripts.ingest_qasper` clears the ChromaDB collection before ingesting.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--papers", type=int, default=N_PAPERS)
    parser.add_argument("--answerable", type=int, default=N_ANSWERABLE)
    parser.add_argument("--unanswerable", type=int, default=N_UNANSWERABLE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    download_dev_split()
    with open(DEV_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    rng = random.Random(args.seed)
    questions = select_questions(raw, rng, args.answerable, args.unanswerable)
    questions.sort(key=lambda q: q["id"])
    corpus = build_corpus(raw, questions, rng, args.papers)

    n_gold = sum(1 for p in corpus.values() if p["is_gold"])
    meta = {
        "dataset": "qasper-v0.3",
        "split": "dev",
        "seed": args.seed,
        "n_documents": len(corpus),
        "n_gold_documents": n_gold,
        "n_distractors": len(corpus) - n_gold,
        "n_questions": len(questions),
        "n_answerable": sum(q["answerable"] for q in questions),
        "n_unanswerable": sum(not q["answerable"] for q in questions),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    with open(OUTPUT_DIR / "golden_set.json", "w", encoding="utf-8") as f:
        json.dump(
            {**meta, "questions": questions}, f, indent=2, ensure_ascii=False,
        )
    write_readme(OUTPUT_DIR / "README.md", meta)

    print(f"Wrote {OUTPUT_DIR}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
