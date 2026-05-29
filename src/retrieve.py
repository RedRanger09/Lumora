from __future__ import annotations
from openai import OpenAI
import os
from dotenv import load_dotenv
import google.generativeai as genai

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR

load_dotenv()

lm_client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    model = None

def generate_answer(question: str, results: list) -> str:
    """Generate grounded answer using retrieved chunks."""

    if model is None:
        return "Gemini API key not configured."

    context = "\n\n".join(
        [
            f"[Source {r.rank}] {r.text}"
            for r in results[:3]
        ]
    )

    prompt = f"""
You are an academic study assistant.

Answer ONLY using the provided context.

If the answer is not present, say:
'I could not find relevant information in the knowledge base.'

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:

        print(f"[Gemini failed: {e}]")
        print("[Falling back to LM Studio...]")

        try:

            local_response = lm_client.chat.completions.create(
                model="local-model",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an academic study assistant. Answer ONLY using provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
            )

            return (
                local_response
                .choices[0]
                .message
                .content
                .strip()
            )

        except Exception as local_e:
            return f"Local generation failed: {local_e}"


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    base_score: float
    boosted_score: float
    chunk_id: str
    subject: str
    unit: str | None
    file_name: str
    heading: str | None
    chunk_title: str | None
    text: str
    diagram: str | None


def _resolve_vectorstore_paths(vectorstore_dir: Path) -> tuple[Path, Path]:
    index_path = vectorstore_dir / "faiss.index"
    metadata_path = vectorstore_dir / "metadata.jsonl"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Vectorstore artifacts not found. Expected:\n"
            f"- {index_path}\n"
            f"- {metadata_path}\n"
            "Run embedding first: python -m src.embed"
        )
    return index_path, metadata_path


def load_index(vectorstore_dir: Path = VECTORSTORE_DIR) -> faiss.Index:
    """Load FAISS index from disk."""
    index_path, _ = _resolve_vectorstore_paths(vectorstore_dir)
    return faiss.read_index(str(index_path))


def load_metadata(vectorstore_dir: Path = VECTORSTORE_DIR) -> list[dict[str, Any]]:
    """Load vector-aligned metadata from JSONL."""
    _, metadata_path = _resolve_vectorstore_paths(vectorstore_dir)
    metadata: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))
    if not metadata:
        raise ValueError(f"Metadata file is empty: {metadata_path}")
    return metadata


def _load_model_name(vectorstore_dir: Path) -> str:
    """Read model from manifest if present, else fallback to config default."""
    manifest_path = vectorstore_dir / "manifest.json"
    if not manifest_path.exists():
        return EMBEDDING_MODEL
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return EMBEDDING_MODEL
    return str(manifest.get("model_name", EMBEDDING_MODEL))


def _load_chunks_file_from_manifest(vectorstore_dir: Path) -> Path | None:
    manifest_path = vectorstore_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source = manifest.get("source_chunks_file")
    if not source:
        return None
    chunks_path = Path(str(source))
    return chunks_path if chunks_path.exists() else None


def _load_chunk_text_map(vectorstore_dir: Path) -> dict[str, str]:
    """Load chunk_id -> text map to print retrieved chunk content."""
    chunks_path = _load_chunks_file_from_manifest(vectorstore_dir)
    if not chunks_path:
        return {}

    text_map: dict[str, str] = {}
    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id", ""))
            text = str(row.get("text", ""))
            if chunk_id:
                text_map[chunk_id] = text
    return text_map


def build_query_encoder(vectorstore_dir: Path = VECTORSTORE_DIR) -> SentenceTransformer:
    """Load the same embedding model used for indexing."""
    model_name = _load_model_name(vectorstore_dir)
    return SentenceTransformer(model_name)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Embed one query and normalize for cosine/IP search."""
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vec, dtype=np.float32)


def _query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(term) > 1]


def _overlap_ratio(query: str, candidate: str | None) -> float:
    if not candidate:
        return 0.0
    q_terms = set(_query_terms(query))
    if not q_terms:
        return 0.0
    c_terms = set(_query_terms(candidate))
    if not c_terms:
        return 0.0
    return len(q_terms.intersection(c_terms)) / len(q_terms)


def _boost_score(query: str, base_score: float, row: dict[str, Any]) -> float:
    """Boost chunks whose heading/title directly matches the query intent."""
    heading = str(row.get("heading", "") or "")
    title = str(row.get("chunk_title", "") or "")
    heading_boost = 0.20 * _overlap_ratio(query, heading)
    title_boost = 0.12 * _overlap_ratio(query, title)
    exact_heading_bonus = 0.15 if heading and query.lower() in heading.lower() else 0.0
    return base_score + heading_boost + title_boost + exact_heading_bonus


def retrieve_with_metadata(
    query: str,
    *,
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[SearchResult]:
    """Retrieve top-k chunks and join vector scores with metadata."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not query.strip():
        return []

    query_vector = embed_query(query, model=model)
    # Pull a wider candidate set from FAISS, then rerank with metadata boosts.
    limit = min(max(top_k * 5, top_k + 10), len(metadata))
    scores, indices = index.search(query_vector, limit)

    scored_candidates: list[tuple[float, float, dict[str, Any]]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[idx]
        base_score = float(score)
        boosted = _boost_score(query, base_score, row)
        scored_candidates.append((boosted, base_score, row))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    winners = scored_candidates[:top_k]

    results: list[SearchResult] = []
    for rank, (boosted_score, base_score, row) in enumerate(winners, start=1):
        results.append(
            SearchResult(
                rank=rank,
                score=boosted_score,
                base_score=base_score,
                boosted_score=boosted_score,
                chunk_id=str(row.get("chunk_id", "")),
                subject=str(row.get("subject", "unknown")),
                unit=row.get("unit"),
                file_name=str(row.get("file_name", "")),
                heading=row.get("heading"),
                chunk_title=row.get("chunk_title"),
                text=str(row.get("text", "")),
                diagram=row.get("diagram"),
            )
        )
    return results


def _format_preview(text: str, max_chars: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."



def has_relevant_results(results: list[SearchResult], min_score: float = 0.64) -> bool:
    """Check if retrieved results pass confidence threshold."""
    return bool(results) and results[0].score >= min_score


def print_results(results: list[SearchResult]) -> None:
    """Pretty-print retrieval matches for terminal evaluation."""

    MIN_SCORE = 0.64

    if not results:
        print("No matches found.")
        return

    if results[0].score < MIN_SCORE:
        print("\nNo relevant information found in knowledge base.")
        print(f"(top score={results[0].score:.4f} below threshold {MIN_SCORE})")
        return

    for item in results:
        unit = item.unit or "no_unit"
        print(f"[{item.rank}] score={item.score:.4f}")
        print(f"    base_score={item.base_score:.4f} | boosted_score={item.boosted_score:.4f}")
        print(f"    subject={item.subject} | unit={unit} | file={item.file_name}")
        if item.heading:
            print(f"    heading={item.heading}")
        if item.chunk_title and item.chunk_title != item.heading:
            print(f"    title={item.chunk_title}")
        print(f"    chunk_id={item.chunk_id}")
        print(f"    preview={_format_preview(item.text)}")
        print()


def run_query_once(query: str, vectorstore_dir: Path, top_k: int) -> list[SearchResult]:
    """Execute one retrieval query and return matches."""
    index = load_index(vectorstore_dir)
    metadata = load_metadata(vectorstore_dir)
    text_map = _load_chunk_text_map(vectorstore_dir)
    if text_map:
        metadata = [
            {**row, "text": text_map.get(str(row.get("chunk_id", "")), str(row.get("text", "")))}
            for row in metadata
        ]
    model = build_query_encoder(vectorstore_dir)
    return retrieve_with_metadata(
        query=query,
        index=index,
        metadata=metadata,
        model=model,
        top_k=top_k,
    )


def interactive_loop(vectorstore_dir: Path, top_k: int) -> int:
    """Interactive terminal mode for iterative retrieval testing."""
    index = load_index(vectorstore_dir)
    metadata = load_metadata(vectorstore_dir)
    text_map = _load_chunk_text_map(vectorstore_dir)
    if text_map:
        metadata = [
            {**row, "text": text_map.get(str(row.get("chunk_id", "")), str(row.get("text", "")))}
            for row in metadata
        ]
    model = build_query_encoder(vectorstore_dir)
    print("Retrieval test mode. Type your question (or 'exit' to quit).")

    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting retrieval test mode.")
            return 0
        if not query:
            print("Please enter a non-empty question.")
            continue
        results = retrieve_with_metadata(
            query=query,
            index=index,
            metadata=metadata,
            model=model,
            top_k=top_k,
        )
        print_results(results)
        if has_relevant_results(results):
            print("\nGenerating answer...\n")

            answer = generate_answer(query, results)

            print("Answer:")
            print(answer)

            print("\nSources:")
            for r in results[:3]:
                unit = r.unit or "unknown"
                print(f"- {r.subject} | {unit} | {r.chunk_id}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG retrieval + Gemini answer generation.")
    parser.add_argument("--query", help="Run one retrieval query and exit.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vectorstore-dir", type=Path, default=VECTORSTORE_DIR)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.query:
        results = run_query_once(
            query=args.query,
            vectorstore_dir=args.vectorstore_dir,
            top_k=args.top_k,
        )
        print_results(results)
        if has_relevant_results(results):
            print("\nGenerating answer...\n")
            answer = generate_answer(args.query, results)
            print("Answer:")
            print(answer)

            print("\nSources:")
            for r in results[:3]:
                unit = r.unit or "unknown"
                print(f"- {r.subject} | {unit} | {r.chunk_id}")
        return 0

    return interactive_loop(vectorstore_dir=args.vectorstore_dir, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())