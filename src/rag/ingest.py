"""Turn the downloaded arXiv PDFs into chunks stored in a persistent Chroma collection.

Run with `python -m src.rag.ingest` after `python -m src.rag.corpus_fetch`. Every run drops and
rebuilds the collection, so it is safe to re-run. Embeddings use Chroma's default local ONNX
model (all-MiniLM-L6-v2); no API key is needed.
"""

import bisect
import json
import logging
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings

logger = logging.getLogger(__name__)
# pypdf logs a warning for every unusual font encoding; they are harmless for text extraction.
logging.getLogger("pypdf").setLevel(logging.ERROR)

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
# Paragraph breaks first, then lines, then sentences, then words.
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MIN_CHUNK_CHARS = 80
ADD_BATCH_SIZE = 256

# A line repeated on at least this share of pages is a running header/footer.
REPEATED_LINE_PAGE_SHARE = 0.5
MAX_HEADING_CHARS = 80
MAX_HEADING_WORDS = 12

REFERENCES_HEADING = re.compile(
    r"^(?:\d+\.?\s*)?(?:references|bibliography|references and notes)$", re.IGNORECASE
)
# Splits a heading into its label and text: "3.2 Retrieval", "4. EXPERIMENTS", "II. OVERVIEW",
# "A Additional Results", "B.2. Prompts". Case-sensitive on purpose: real headings start uppercase.
HEADING_LABEL = re.compile(
    r"^(?P<label>\d{1,2}(?:\.\d{1,2}){0,2}|[IVX]{1,5}|[A-H](?:\.\d{1,2}){0,2})(?P<dot>\.?)\s+(?P<text>[A-Z].*)$"
)
ROMAN_VALUES = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
# "Appendix A ...", "APPENDIX", "Supplementary Material"
APPENDIX_WORD_HEADING = re.compile(r"^(?:appendix|appendices|supplementary material)\b", re.IGNORECASE)
# Affiliation lines ("1 Department of ...") are numbered like sections; never treat them as headings.
AFFILIATION_WORDS = re.compile(r"universit|department|school of|institute|laborator|@|\.com|\.edu", re.IGNORECASE)
# Table-of-contents entries end with a page number: "2 Background 6".
TRAILING_PAGE_NUMBER = re.compile(r"\s\d+$")
# Headings trusted anywhere in the paper.
ALWAYS_HEADINGS = {"abstract", "acknowledgements", "acknowledgments", "appendix"}
# Headings trusted only in papers without numbered sections.
UNNUMBERED_HEADINGS = {
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
}
# pypdf renders small caps as "I NTRODUCTION"; re-join a lone capital with the capitals after it.
SMALL_CAPS_GAP = re.compile(r"\b([A-Z]) (?=[A-Z]{2,})")
SENTENCE_END = (".", "!", "?", ":")


@dataclass
class PageText:
    page_number: int  # 1-based
    lines: list[str]


@dataclass
class CleanDocument:
    text: str
    page_offsets: list[int] = field(default_factory=list)  # char offset where each page starts
    page_numbers: list[int] = field(default_factory=list)
    section_offsets: list[int] = field(default_factory=list)  # char offset where each heading starts
    section_names: list[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict


@dataclass
class IngestStats:
    papers: int
    chunks: int
    pages: int
    seconds: float


def load_metadata(corpus_dir: Path) -> list[dict]:
    metadata_path = corpus_dir / "metadata.json"
    raw_text = metadata_path.read_text()
    return json.loads(raw_text)


def read_pages(pdf_path: Path) -> list[PageText]:
    reader = PdfReader(pdf_path)
    pages: list[PageText] = []
    for page_index, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception as extract_error:
            logger.warning("page %d of %s unreadable: %s", page_index + 1, pdf_path.name, extract_error)
            raw_text = ""
        # NFKC folds ligatures such as "ﬁ" into "fi".
        normalised = unicodedata.normalize("NFKC", raw_text)
        stripped_lines = [line.strip() for line in normalised.split("\n")]
        pages.append(PageText(page_number=page_index + 1, lines=stripped_lines))
    return pages


def find_repeated_lines(pages: list[PageText]) -> set[str]:
    """Running headers/footers: short lines that show up on most pages."""
    if len(pages) < 4:
        return set()
    line_page_counts: Counter[str] = Counter()
    for page in pages:
        unique_lines = {line for line in page.lines if line and len(line) < 120}
        line_page_counts.update(unique_lines)
    threshold = len(pages) * REPEATED_LINE_PAGE_SHARE
    return {line for line, count in line_page_counts.items() if count >= threshold}


def is_noise_line(line: str, repeated_lines: set[str]) -> bool:
    if not line:
        return True
    if line in repeated_lines:
        return True
    # Bare page numbers.
    return line.isdigit()


def repair_small_caps(text: str) -> str:
    return SMALL_CAPS_GAP.sub(r"\1", text)


def looks_like_heading_text(text: str) -> bool:
    if len(text) > MAX_HEADING_CHARS or text.endswith((".", ",", ";", ":")):
        return False
    if AFFILIATION_WORDS.search(text) or TRAILING_PAGE_NUMBER.search(text):
        return False
    words = text.split()
    if not words or len(words) > MAX_HEADING_WORDS:
        return False
    letter_count = sum(1 for character in text if character.isalpha())
    # Table rows like "2 GPT-4 85.2 71.0" are mostly digits; real headings are mostly letters.
    return letter_count >= 0.7 * len(text.replace(" ", ""))


@dataclass
class HeadingLabel:
    kind: str  # "arabic", "roman" or "letter"
    parts: list[str]  # "3.2" -> ["3", "2"]
    has_dot: bool  # "4. Experiments" vs "4 Experiments"
    text: str


def parse_heading_label(line: str) -> HeadingLabel | None:
    label_match = HEADING_LABEL.match(line)
    if label_match is None:
        return None
    label = label_match.group("label")
    heading_text = repair_small_caps(label_match.group("text"))
    if not looks_like_heading_text(heading_text):
        return None
    if label[0].isdigit():
        kind = "arabic"
    elif label in ROMAN_VALUES:
        kind = "roman"
    else:
        kind = "letter"
    # Single "I", "V", "X" could be roman or letters; letters only go up to H, so treat them as roman.
    return HeadingLabel(kind=kind, parts=label.split("."), has_dot=bool(label_match.group("dot")), text=heading_text)


@dataclass
class HeadingTracker:
    """Numbering seen so far. A candidate heading is accepted only if it continues the sequence,
    which filters out list items, figure text and affiliations that look like headings."""

    style: str = ""  # "arabic", "arabic_dot" or "roman", fixed by the first accepted section
    section_number: int = 0
    sub_letter: str = ""  # roman-style papers use "A. ...", "B. ..." for subsections
    appendix_letter: str = ""


def next_letter(letter: str) -> str:
    if not letter:
        return "A"
    return chr(ord(letter) + 1)


def accept_section_number(top_number: int, is_subsection: bool, tracker: HeadingTracker) -> bool:
    # Sections advance by one ("3" after "2"); subsections stay inside the current section
    # ("3.2" while in 3) or open the next one when its own heading line was missed.
    if is_subsection:
        return top_number in (tracker.section_number, tracker.section_number + 1)
    return top_number == tracker.section_number + 1


def accept_arabic(label: HeadingLabel, tracker: HeadingTracker) -> bool:
    style = "arabic_dot" if label.has_dot and len(label.parts) == 1 else "arabic"
    is_subsection = len(label.parts) > 1
    # The first section fixes the style; "2. Increasing ..." in a figure of a "1 Introduction"
    # paper is then rejected.
    if not is_subsection and tracker.style and style != tracker.style:
        return False
    if tracker.style == "roman":
        return False
    if not accept_section_number(int(label.parts[0]), is_subsection, tracker):
        return False
    if not tracker.style and not is_subsection:
        tracker.style = style
    tracker.section_number = int(label.parts[0])
    return True


def accept_roman(label: HeadingLabel, tracker: HeadingTracker) -> bool:
    if tracker.style not in ("", "roman") or not label.has_dot:
        return False
    top_number = ROMAN_VALUES[label.parts[0]]
    if not accept_section_number(top_number, False, tracker):
        return False
    tracker.style = "roman"
    tracker.section_number = top_number
    tracker.sub_letter = ""
    return True


def accept_letter(label: HeadingLabel, tracker: HeadingTracker) -> bool:
    letter = label.parts[0]
    is_subsection = len(label.parts) > 1
    if tracker.section_number == 0:
        return False
    # Roman-numbered papers use "A.", "B." as subsections inside each section.
    if tracker.style == "roman" and label.has_dot and not is_subsection:
        if letter != next_letter(tracker.sub_letter):
            return False
        tracker.sub_letter = letter
        return True
    # Otherwise letters are appendix sections and run A, B, C, ...
    if is_subsection:
        accepted = letter in (tracker.appendix_letter, next_letter(tracker.appendix_letter))
    else:
        accepted = letter == next_letter(tracker.appendix_letter)
    if accepted:
        tracker.appendix_letter = letter
    return accepted


def detect_heading(line: str, tracker: HeadingTracker) -> str | None:
    """Return a normalised heading if the line looks like a section heading, else None."""
    lowered = repair_small_caps(line).lower()
    if lowered in ALWAYS_HEADINGS:
        return lowered.title()
    # Words like "Method" also appear as table column headers, so only trust bare words
    # in papers that do not number their sections.
    if lowered in UNNUMBERED_HEADINGS and tracker.section_number == 0:
        return lowered.title()
    label = parse_heading_label(line)
    if label is None:
        return None
    acceptors = {"arabic": accept_arabic, "roman": accept_roman, "letter": accept_letter}
    accept_label = acceptors[label.kind]
    if not accept_label(label, tracker):
        return None
    label_text = ".".join(label.parts)
    dot_text = "." if label.has_dot else ""
    return f"{label_text}{dot_text} {label.text}"


def is_references_heading(line: str) -> bool:
    candidate = repair_small_caps(line)
    return bool(REFERENCES_HEADING.match(candidate))


def is_appendix_heading(line: str) -> bool:
    if "," in line:
        return False
    if APPENDIX_WORD_HEADING.match(line):
        return len(line) <= MAX_HEADING_CHARS
    # The first appendix section is "A ...", "A. ..." or "A.1 ...". Case-sensitive (see
    # HEADING_LABEL) so a wrapped reference line like "a survey of ..." never ends the bibliography.
    label = parse_heading_label(line)
    return label is not None and label.kind == "letter" and label.parts[0] == "A"


def join_page_lines(
    lines: list[str], typical_line_length: float, tracker: HeadingTracker
) -> list[tuple[str, bool]]:
    """Merge wrapped lines into paragraphs.

    Returns (paragraph_text, is_heading) pairs. A paragraph ends at a heading or at a short line
    that finishes a sentence (the ragged last line of a paragraph).
    """
    paragraphs: list[tuple[str, bool]] = []
    current_parts: list[str] = []

    def flush_paragraph() -> None:
        if current_parts:
            paragraphs.append((" ".join(current_parts), False))
            current_parts.clear()

    for line in lines:
        heading = detect_heading(line, tracker)
        if heading is not None:
            flush_paragraph()
            paragraphs.append((heading, True))
            continue
        # De-hyphenate words split across a line break: "Con-" + "sider" -> "Consider".
        if current_parts and current_parts[-1].endswith("-") and line[:1].islower():
            current_parts[-1] = current_parts[-1][:-1] + line
        else:
            current_parts.append(line)
        if line.endswith(SENTENCE_END) and len(line) < 0.8 * typical_line_length:
            flush_paragraph()
    flush_paragraph()
    return paragraphs


def collapse_whitespace(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", text).strip()


def build_clean_document(pages: list[PageText]) -> CleanDocument:
    repeated_lines = find_repeated_lines(pages)
    document = CleanDocument(text="")
    text_parts: list[str] = []
    cursor = 0
    in_references = False
    tracker = HeadingTracker()

    for page in pages:
        content_lines = [line for line in page.lines if not is_noise_line(line, repeated_lines)]
        kept_lines: list[str] = []
        for line in content_lines:
            # Drop the bibliography, but resume if an appendix follows it.
            if not in_references and is_references_heading(line):
                in_references = True
                continue
            if in_references:
                if is_appendix_heading(line):
                    in_references = False
                    kept_lines.append(line)
                continue
            kept_lines.append(line)
        if not kept_lines:
            continue

        line_lengths = [len(line) for line in kept_lines]
        typical_line_length = statistics.median(line_lengths)
        paragraphs = join_page_lines(kept_lines, typical_line_length, tracker)

        document.page_offsets.append(cursor)
        document.page_numbers.append(page.page_number)
        for paragraph_text, is_heading in paragraphs:
            cleaned_paragraph = collapse_whitespace(paragraph_text)
            if not cleaned_paragraph:
                continue
            if is_heading:
                document.section_offsets.append(cursor)
                document.section_names.append(cleaned_paragraph)
            text_parts.append(cleaned_paragraph)
            # Paragraphs are separated by a blank line so the splitter prefers to cut there.
            cursor += len(cleaned_paragraph) + 2

    document.text = "\n\n".join(text_parts)
    return document


def lookup_at_offset(offsets: list[int], values: list, offset: int, default):
    # Last entry whose start offset is <= the chunk start.
    position = bisect.bisect_right(offsets, offset) - 1
    if position < 0:
        return default
    return values[position]


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        add_start_index=True,
        # Keep ". " on the end of the previous piece so chunks never start with a stray period.
        keep_separator="end",
    )


def chunk_paper(paper: dict, pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> tuple[list[ChunkRecord], int]:
    pages = read_pages(pdf_path)
    document = build_clean_document(pages)
    if not document.text:
        return [], len(pages)

    split_documents = splitter.create_documents([document.text])
    records: list[ChunkRecord] = []
    for split_document in split_documents:
        chunk_text = split_document.page_content.strip()
        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue
        start_offset = split_document.metadata["start_index"]
        page_number = lookup_at_offset(document.page_offsets, document.page_numbers, start_offset, 1)
        section_name = lookup_at_offset(document.section_offsets, document.section_names, start_offset, "")
        chunk_index = len(records)
        records.append(
            ChunkRecord(
                chunk_id=f"{paper['versioned_id']}::chunk-{chunk_index:04d}",
                text=chunk_text,
                metadata={
                    "source_id": paper["id"],
                    "title": paper["title"],
                    "section": section_name,
                    "page": int(page_number),
                    "url": paper["url"],
                    "chunk_index": chunk_index,
                },
            )
        )
    return records, len(pages)


def open_client(persist_dir: str) -> chromadb.ClientAPI:
    client_settings = ChromaSettings(anonymized_telemetry=False)
    return chromadb.PersistentClient(path=persist_dir, settings=client_settings)


def recreate_collection(client: chromadb.ClientAPI, collection_name: str) -> chromadb.Collection:
    existing_names = [collection.name for collection in client.list_collections()]
    if collection_name in existing_names:
        client.delete_collection(collection_name)
    return client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=DefaultEmbeddingFunction(),
    )


def add_in_batches(collection: chromadb.Collection, records: list[ChunkRecord]) -> None:
    for batch_start in range(0, len(records), ADD_BATCH_SIZE):
        batch = records[batch_start : batch_start + ADD_BATCH_SIZE]
        collection.add(
            ids=[record.chunk_id for record in batch],
            documents=[record.text for record in batch],
            metadatas=[record.metadata for record in batch],
        )


def ingest_corpus(
    corpus_dir: str | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> IngestStats:
    started_at = time.perf_counter()
    corpus_path = Path(corpus_dir or settings.corpus_dir)
    papers = load_metadata(corpus_path)
    splitter = make_splitter()

    all_records: list[ChunkRecord] = []
    total_pages = 0
    ingested_papers = 0
    for paper in papers:
        pdf_path = corpus_path / paper["pdf_file"]
        if not pdf_path.exists():
            logger.warning("missing PDF for %s; run corpus_fetch first", paper["id"])
            continue
        paper_records, page_count = chunk_paper(paper, pdf_path, splitter)
        total_pages += page_count
        if not paper_records:
            logger.warning("no text extracted from %s", paper["id"])
            continue
        ingested_papers += 1
        all_records.extend(paper_records)
        logger.info("%s: %d pages -> %d chunks", paper["id"], page_count, len(paper_records))

    client = open_client(persist_dir or settings.chroma_persist_dir)
    collection = recreate_collection(client, collection_name or settings.chroma_collection)
    add_in_batches(collection, all_records)

    elapsed_seconds = time.perf_counter() - started_at
    stats = IngestStats(
        papers=ingested_papers,
        chunks=collection.count(),
        pages=total_pages,
        seconds=round(elapsed_seconds, 1),
    )
    logger.info(
        "ingest done: %d papers, %d pages, %d chunks in %.1fs",
        stats.papers,
        stats.pages,
        stats.chunks,
        stats.seconds,
    )
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest_corpus()


if __name__ == "__main__":
    main()
