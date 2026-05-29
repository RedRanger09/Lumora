from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator



from src.config import CHUNKS_DIR, OCR_TEXT_DIR
DIAGRAM_MAP_PATH = Path("data/diagrams/diagram_map.json")
WORD_PATTERN = re.compile(r"\S+")
UNIT_PATTERN = re.compile(r"unit\d+", re.IGNORECASE)
HEADING_HINT_RE = re.compile(
    r"^(unit[-\s]?\d+|chapter\s+\d+|module\s+\d+|topic|syllabus|important questions?|\d+\s*[\).:\-]\s*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    subject: str
    unit: str | None
    file_name: str
    source_path: str
    word_count: int
    chunk_index: int
    total_chunks_in_file: int
    source_files: list[str]
    heading: str | None
    chunk_title: str | None
    diagram: str | None


@dataclass(frozen=True)
class TextFileInfo:
    path: Path
    subject: str
    unit: str | None
    file_name: str


@dataclass(frozen=True)
class ParagraphBlock:
    text: str
    subject: str
    unit: str | None
    file_name: str
    source_path: str
    is_heading: bool
    heading: str | None


def count_words(text: str) -> int:
    """Count words robustly for mixed OCR text."""
    return len(WORD_PATTERN.findall(text))


def normalize_text(text: str) -> str:
    """Normalize OCR text while keeping paragraph boundaries."""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs based on blank lines."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"\n\s*\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def is_heading_like(paragraph: str) -> bool:
    """Detect heading/topic boundaries to avoid unrelated merges."""
    compact = " ".join(paragraph.split())
    words = compact.split()
    if not words:
        return False
    if len(words) <= 10 and (compact.isupper() or compact.endswith(":")):
        return True
    if len(words) <= 14 and re.match(r"^\d+\s*[).:\-]\s*", compact):
        return True
    if len(words) <= 10 and compact.istitle():
        return True
    if HEADING_HINT_RE.search(compact):
        return True
    return False


def _sentence_split(paragraph: str) -> list[str]:
    """Fallback semantic splitter when a paragraph is too long."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _windowed_word_chunks(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Last-resort splitter for very long OCR blocks with weak punctuation."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + max_words
        slice_words = words[start:end]
        if not slice_words:
            continue
        chunks.append(" ".join(slice_words))
        if end >= len(words):
            break
    return chunks


def _split_overlong_paragraph(paragraph: str, max_words: int, overlap_words: int) -> list[str]:
    """Split an oversized paragraph semantically before fixed windows."""
    if count_words(paragraph) <= max_words:
        return [paragraph]

    sentences = _sentence_split(paragraph)
    if len(sentences) <= 1:
        return _windowed_word_chunks(paragraph, max_words=max_words, overlap_words=overlap_words)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = count_words(sentence)
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _tail_words(text: str, overlap_words: int) -> str:
    """Return trailing words for overlap carry-over."""
    if overlap_words <= 0:
        return ""
    words = text.split()
    if not words:
        return ""
    return " ".join(words[-overlap_words:])


def build_chunks_from_paragraphs(
    paragraphs: list[ParagraphBlock],
    target_min_words: int = 250,
    target_max_words: int = 400,
    overlap_words: int = 50,
) -> list[list[ParagraphBlock]]:
    """Pack paragraph blocks into topic-aware chunks near 250-400 words."""
    if target_min_words <= 0 or target_max_words <= 0:
        raise ValueError("target_min_words and target_max_words must be positive")
    if target_min_words > target_max_words:
        raise ValueError("target_min_words must be <= target_max_words")

    expanded: list[ParagraphBlock] = []
    for paragraph in paragraphs:
        split_parts = _split_overlong_paragraph(paragraph.text, target_max_words, overlap_words)
        expanded.extend(
            ParagraphBlock(
                text=part,
                subject=paragraph.subject,
                unit=paragraph.unit,
                file_name=paragraph.file_name,
                source_path=paragraph.source_path,
                is_heading=paragraph.is_heading,
                heading=paragraph.heading,
            )
            for part in split_parts
        )

    if not expanded:
        return []

    chunks: list[list[ParagraphBlock]] = []
    current_parts: list[ParagraphBlock] = []
    current_words = 0

    for paragraph in expanded:
        paragraph_words = count_words(paragraph.text)
        if paragraph_words == 0:
            continue

        # Always start a new chunk at a new heading to preserve topic boundaries.
        if current_parts and paragraph.is_heading:
            chunks.append(current_parts)
            overlap_prefix = _tail_words("\n\n".join(item.text for item in current_parts), overlap_words)
            current_parts = (
                [
                    ParagraphBlock(
                        overlap_prefix,
                        paragraph.subject,
                        paragraph.unit,
                        paragraph.file_name,
                        paragraph.source_path,
                        False,
                        paragraph.heading,
                    ),
                    paragraph,
                ]
                if overlap_prefix
                else [paragraph]
            )
            current_words = count_words("\n\n".join(item.text for item in current_parts))
            continue

        would_exceed = current_words + paragraph_words > target_max_words
        if current_parts and would_exceed:
            chunks.append(current_parts)
            completed_text = "\n\n".join(item.text for item in current_parts).strip()
            overlap_prefix = _tail_words(completed_text, overlap_words)
            current_parts = (
                [
                    ParagraphBlock(
                        overlap_prefix,
                        paragraph.subject,
                        paragraph.unit,
                        paragraph.file_name,
                        paragraph.source_path,
                        False,
                        paragraph.heading,
                    ),
                    paragraph,
                ]
                if overlap_prefix
                else [paragraph]
            )
            current_words = count_words("\n\n".join(item.text for item in current_parts))
            continue

        current_parts.append(paragraph)
        current_words += paragraph_words

        if current_words >= target_min_words:
            chunks.append(current_parts)
            completed_text = "\n\n".join(item.text for item in current_parts).strip()
            overlap_prefix = _tail_words(completed_text, overlap_words)
            current_parts = (
                [
                    ParagraphBlock(
                        overlap_prefix,
                        paragraph.subject,
                        paragraph.unit,
                        paragraph.file_name,
                        paragraph.source_path,
                        False,
                        paragraph.heading,
                    )
                ]
                if overlap_prefix
                else []
            )
            current_words = count_words(overlap_prefix) if overlap_prefix else 0

    if current_parts:
        tail = "\n\n".join(item.text for item in current_parts).strip()
        if tail:
            if chunks and count_words(tail) < max(120, target_min_words // 3):
                chunks[-1] = chunks[-1] + current_parts
            else:
                chunks.append(current_parts)

    normalized_chunks: list[list[ParagraphBlock]] = []
    for chunk in chunks:
        if isinstance(chunk, list):
            if count_words("\n\n".join(item.text for item in chunk)) > 0:
                normalized_chunks.append(chunk)
        else:
            # safety for old path, should not happen
            if count_words(str(chunk)) > 0:
                normalized_chunks.append(
                    [ParagraphBlock(str(chunk), "unknown", None, "", "", False, None)]
                )
    return normalized_chunks


def parse_metadata_from_path(text_path: Path, input_dir: Path) -> TextFileInfo:
    """Derive subject/unit/file metadata from OCR text file path."""
    relative = text_path.relative_to(input_dir)
    parts = relative.parts
    subject = parts[0] if parts else "unknown"
    unit: str | None = None
    if len(parts) > 1 and UNIT_PATTERN.fullmatch(parts[1] or ""):
        unit = parts[1].lower()
    return TextFileInfo(
        path=text_path,
        subject=subject.lower(),
        unit=unit,
        file_name=text_path.name,
    )


def iter_text_files(input_dir: Path, subject: str | None = None) -> Iterator[Path]:
    """Yield OCR text files in stable order."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if subject:
        root = input_dir / subject
        if not root.exists():
            raise FileNotFoundError(f"Subject directory not found: {root}")
        roots = [root]
    else:
        roots = sorted(path for path in input_dir.iterdir() if path.is_dir())

    for root in roots:
        for file_path in sorted(root.rglob("*.txt")):
            if file_path.is_file():
                yield file_path


def _group_by_subject_unit(paths: list[Path], input_dir: Path) -> dict[tuple[str, str | None], list[Path]]:
    grouped: dict[tuple[str, str | None], list[Path]] = {}
    for path in paths:
        info = parse_metadata_from_path(path, input_dir)
        key = (info.subject, info.unit)
        grouped.setdefault(key, []).append(path)
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    return grouped
def load_diagram_map() -> dict:
    if not DIAGRAM_MAP_PATH.exists():
        return {}

    with DIAGRAM_MAP_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


DIAGRAM_MAP = load_diagram_map()


def find_diagram_for_heading(heading: str | None) -> str | None:
    if not heading:
        return None

    heading_clean = re.sub(
        r"[^a-z0-9\s]",
        " ",
        heading.lower()
    )

    heading_clean = " ".join(heading_clean.split())

    for key, path in DIAGRAM_MAP.items():

        key_clean = re.sub(
            r"[^a-z0-9\s]",
            " ",
            key.lower()
        )

        key_clean = " ".join(key_clean.split())

        if key_clean in heading_clean:
            return str(path)

    return None


def _build_paragraph_blocks_for_file(text_path: Path, input_dir: Path) -> list[ParagraphBlock]:
    info = parse_metadata_from_path(text_path, input_dir)
    raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = split_into_paragraphs(raw_text)
    blocks: list[ParagraphBlock] = []
    active_heading: str | None = None
    for paragraph in paragraphs:
        heading_flag = is_heading_like(paragraph)
        if heading_flag:
            active_heading = paragraph
        blocks.append(
            ParagraphBlock(
                text=paragraph,
                subject=info.subject,
                unit=info.unit,
                file_name=info.file_name,
                source_path=str(text_path),
                is_heading=heading_flag,
                heading=active_heading,
            )
        )
    return blocks


def _select_chunk_heading(block_group: list[ParagraphBlock]) -> str | None:
    for block in block_group:
        if block.is_heading:
            return block.text.strip()
    for block in block_group:
        if block.heading:
            return block.heading.strip()
    return None


def _derive_chunk_title(block_group: list[ParagraphBlock], heading: str | None) -> str:
    if heading:
        return heading
    for block in block_group:
        text = " ".join(block.text.split())
        if not text:
            continue
        words = text.split()
        return " ".join(words[:10]).strip()
    return "untitled_chunk"


def chunk_unit_corpus(
    text_paths: list[Path],
    input_dir: Path = OCR_TEXT_DIR,
    target_min_words: int = 250,
    target_max_words: int = 400,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Create chunk objects from multiple files within one subject/unit stream."""
    if not text_paths:
        return []
    first_info = parse_metadata_from_path(text_paths[0], input_dir)
    paragraph_blocks: list[ParagraphBlock] = []
    for text_path in text_paths:
        paragraph_blocks.extend(_build_paragraph_blocks_for_file(text_path, input_dir))

    chunk_blocks = build_chunks_from_paragraphs(
        paragraphs=paragraph_blocks,
        target_min_words=target_min_words,
        target_max_words=target_max_words,
        overlap_words=overlap_words,
    )

    total = len(chunk_blocks)
    chunks: list[Chunk] = []
    for index, block_group in enumerate(chunk_blocks, start=1):
        text = "\n\n".join(block.text for block in block_group).strip()
        source_files = sorted({block.file_name for block in block_group if block.file_name})
        primary_file = source_files[0] if source_files else ""
        heading = _select_chunk_heading(block_group)
        chunk_title = _derive_chunk_title(block_group, heading)
        chunk_id = (
            f"{first_info.subject}_{first_info.unit or 'no_unit'}"
            f"_chunk_{index:03d}"
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                subject=first_info.subject,
                unit=first_info.unit,
                file_name=primary_file,
                source_path=str(text_paths[0]),
                word_count=count_words(text),
                chunk_index=index,
                total_chunks_in_file=total,
                source_files=source_files,
                heading=heading,
                chunk_title=chunk_title,
                diagram=(
                          find_diagram_for_heading(heading)
                         or find_diagram_for_heading(chunk_title)
                         or find_diagram_for_heading(text[:300])
                        ),
            )
        )
    return chunks


def save_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Persist chunk objects to JSONL for embedding pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def chunk_text_files(
    input_dir: Path = OCR_TEXT_DIR,
    output_dir: Path = CHUNKS_DIR,
    subject: str | None = None,
    target_min_words: int = 250,
    target_max_words: int = 400,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Chunk OCR text files by subject/unit and write one JSONL output."""
    all_chunks: list[Chunk] = []
    text_paths = list(iter_text_files(input_dir=input_dir, subject=subject))
    grouped = _group_by_subject_unit(text_paths, input_dir=input_dir)
    for _, unit_paths in grouped.items():
        all_chunks.extend(
            chunk_unit_corpus(
                text_paths=unit_paths,
                input_dir=input_dir,
                target_min_words=target_min_words,
                target_max_words=target_max_words,
                overlap_words=overlap_words,
            )
        )

    output_file = output_dir / "chunks.jsonl"
    save_chunks_jsonl(all_chunks, output_file)
    return all_chunks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk OCR text files into retrieval-ready objects.")
    parser.add_argument("--subject", help="Process only one subject (e.g., os, java).")
    parser.add_argument("--all", action="store_true", help="Process all subject folders.")
    parser.add_argument("--input-dir", type=Path, default=OCR_TEXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--min-words", type=int, default=250)
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--overlap-words", type=int, default=50)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not args.all and not args.subject:
        print("Provide --all or --subject <name>.")
        return 1

    chunks = chunk_text_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        subject=args.subject,
        target_min_words=args.min_words,
        target_max_words=args.max_words,
        overlap_words=args.overlap_words,
    )
    print(f"Created {len(chunks)} chunks at {args.output_dir / 'chunks.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
