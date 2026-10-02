"""Dataset ingestion: download SQuAD v1.1 dev, build a deduplicated passage corpus.

Deterministic and resumable: skips the download if the file already exists and
parses cleanly, and always iterates the dataset in file order.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SQUAD_PATH = DATA_DIR / "squad_dev_v1.1.json"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def download_squad(dest: Path = SQUAD_PATH, url: str = SQUAD_DEV_URL) -> Path:
    """Download the SQuAD v1.1 dev set. Skips if ``dest`` already parses as JSON."""
    if dest.exists():
        try:
            with open(dest, encoding="utf-8") as f:
                json.load(f)
            print(f"Using cached SQuAD dev set: {dest}")
            return dest
        except (json.JSONDecodeError, OSError):
            print(f"Cached file {dest} is corrupt; re-downloading.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SQuAD v1.1 dev set from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


def build_corpus(squad: dict) -> tuple[list[dict], list[dict]]:
    """Build (passages, questions) from a parsed SQuAD file.

    Passages are the article paragraphs, deduplicated by normalized text in
    file order. Each question records the id of its gold passage (the
    paragraph its answers come from).
    """
    passages: list[dict] = []
    seen: dict[str, str] = {}  # normalized text -> passage id
    questions: list[dict] = []

    for article in squad["data"]:
        title = article.get("title", "")
        for para in article["paragraphs"]:
            context = para["context"]
            key = _norm(context)
            if key not in seen:
                pid = f"{title}::{len(passages)}"
                seen[key] = pid
                passages.append({"id": pid, "title": title, "text": context})
            gold_id = seen[key]
            for qa in para["qas"]:
                answers = [a["text"] for a in qa.get("answers", [])]
                questions.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "answers": answers,
                        "gold_passage_id": gold_id,
                    }
                )
    return passages, questions


def ingest(data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Run the full ingestion pipeline. Returns (corpus_path, questions_path)."""
    squad_path = download_squad(data_dir / "squad_dev_v1.1.json")
    with open(squad_path, encoding="utf-8") as f:
        squad = json.load(f)
    passages, questions = build_corpus(squad)
    corpus_path = data_dir / "corpus.jsonl"
    questions_path = data_dir / "questions.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(questions_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Corpus: {len(passages)} passages -> {corpus_path}")
    print(f"Questions: {len(questions)} -> {questions_path}")
    return corpus_path, questions_path


def load_corpus(corpus_path: Path = CORPUS_PATH) -> list[dict]:
    with open(corpus_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(questions_path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(questions_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    ingest()
