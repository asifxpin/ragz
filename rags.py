"""
Agentic RAG (bilingual: English + Arabic)
-----------------------------------------
Expose retrieval as a tool; the agent decides when to call it and can
call it multiple times with refined queries. Better for multi-hop questions.

Four layers matter for a bilingual knowledge base:
  1. Normalizer  - collapses Arabic spelling variants so matching is stable.
  2. Chunker     - splits on Arabic punctuation (؟ ؛ ،) as well as English.
  3. Embedding   - one model that puts both languages in the same vector space.
  4. Vector store- FAISS; dense search itself is language-agnostic.
"""

import re
from pathlib import Path

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
    normalize_unicode,
)
import faiss
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

load_dotenv()


# ============================================================================
# 1. NORMALIZER
# ============================================================================
# Matches any Arabic-script character, so Arabic rules only run when needed.
ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")


def normalize_bilingual_text(text: str) -> str:
    """Normalize mixed English/Arabic text so retrieval matches reliably."""
    # Safe Unicode normalization for the complete mixed-language text.
    text = normalize_unicode(text)

    if ARABIC_PATTERN.search(text):
        text = dediac_ar(text)                  # remove diacritics (harakat)
        text = normalize_alef_ar(text)          # أ إ آ -> ا
        text = normalize_alef_maksura_ar(text)  # ى -> ي

        # Optional: improves matching, but changes feminine endings (ة -> ه).
        text = normalize_teh_marbuta_ar(text)

    # Preserves Arabic/English word order and removes excess whitespace.
    return " ".join(text.split())


# ============================================================================
# 2. CHUNKER
# ============================================================================
# The default separators are ["\n\n", "\n", " ", ""], which know nothing about
# Arabic punctuation. Long Arabic text therefore falls back to splitting on
# spaces, cutting sentences in half. Listing Arabic marks first lets the
# splitter break on real sentence and clause boundaries in both languages.
BILINGUAL_SEPARATORS = [
    "\n\n",  # paragraphs
    "\n",    # lines
    "۔",     # Arabic full stop        (U+06D4)
    "؟",     # Arabic question mark    (U+061F)
    "!",     # exclamation (shared)
    "؛",     # Arabic semicolon        (U+061B)
    ".",     # full stop (shared)
    ";",
    "،",     # Arabic comma            (U+060C)
    ",",
    " ",     # words
    "",      # characters (last resort)
]

# keep_separator="end" leaves the punctuation attached to the sentence it ends.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=BILINGUAL_SEPARATORS,
    keep_separator="end",
)


# ============================================================================
# 3. LOAD + INDEX
# ============================================================================
files_dir = Path(__file__).parent / "files"


def _source_name(path: Path) -> str:
    """Label a file by its path relative to ./files, with forward slashes.

    Subfolders are searched, so a bare filename would make two files of the
    same name in different folders indistinguishable in the retrieved output.
    """
    return path.relative_to(files_dir).as_posix()


def load_seed_documents() -> list[Document]:
    """Load every .txt under ./files so English and Arabic sources share one index.

    The search is recursive, so subfolders such as ./files/_samples are indexed
    too. Only .txt is read, and since disk is now the sole source of knowledge,
    anything in another format is reported rather than skipped in silence --
    which would look identical to retrieval being broken.
    """
    if files_dir.is_dir():
        skipped = sorted(
            _source_name(path)
            for path in files_dir.rglob("*")
            if path.is_file() and path.suffix.lower() != ".txt"
        )
        if skipped:
            logger.warning(
                f"./files: ignoring {len(skipped)} non-.txt file(s): {', '.join(skipped)}. "
                "Only .txt is indexed; convert other formats to .txt to include them."
            )

    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": _source_name(path)},
        )
        for path in sorted(files_dir.rglob("*.txt"))
    ]


def prepare_splits(documents: list[Document]) -> list[Document]:
    """Chunk documents, then index the normalized text.

    The original wording is kept in metadata so the agent reads natural text
    instead of the stripped-down matching form.
    """
    return [
        Document(
            page_content=normalize_bilingual_text(split.page_content),
            metadata={**split.metadata, "original_text": split.page_content},
        )
        for split in splitter.split_documents(documents)
    ]


docs = load_seed_documents()
indexed_splits = prepare_splits(docs)

# text-embedding-3-large places English and Arabic in one shared space, so an
# Arabic question can retrieve an English chunk and vice versa.
# Self-hosted alternative: BAAI/bge-m3 via HuggingFaceEmbeddings.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def build_store(splits: list[Document]) -> FAISS:
    """Build a fresh index, tolerating an empty corpus.

    ./files is allowed to be empty so that a checkout with no documents still
    starts instead of crashing on import. FAISS needs the vector width before it
    can allocate, and ``FAISS.from_documents`` infers that width from the first
    embedding it gets back, so with zero documents it raises IndexError.
    Building the store by hand and taking the width from one probe embedding is
    the way around it.

    IndexFlatL2 with no L2 normalization is exactly what ``from_documents``
    constructs by default, so retrieval behaves the same either way.
    """
    if splits:
        return FAISS.from_documents(splits, embeddings)

    return FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(len(embeddings.embed_query("dimension probe"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


# The index is built once at import and only ever read after that, so the voice
# pipeline is its single caller and no locking is needed.
vectorstore = build_store(indexed_splits)


def search_multilingual(query: str, k: int = 3) -> list[Document]:
    """Search one shared index using English, Arabic, or mixed text."""
    if not query.strip():
        return []

    return vectorstore.similarity_search(
        normalize_bilingual_text(query),
        k=k,
    )


# ============================================================================
# 4. RETRIEVAL TOOL
# ============================================================================
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve knowledge with an English, Arabic, or mixed-language query."""
    results = search_multilingual(query)

    if not results:
        return "No relevant knowledge was found.", []

    text = "\n\n".join(
        f"Source: {d.metadata['source']}\n{d.metadata.get('original_text', d.page_content)}"
        for d in results
    )
    return text, results


# Retrieval smoke check:
#
#   uv run python rags.py
#   uv run python rags.py "your question" "سؤالك بالعربية"
#
# The contents of ./files are not known ahead of time, so this reports what came
# back rather than asserting a specific answer. Ground truth would have to be
# rewritten every time the corpus changes, and a stale expectation that always
# prints FAIL is worse than no expectation at all.
if __name__ == "__main__":
    import sys

    print(f"{files_dir}: {len(indexed_splits)} chunks from {len(docs)} file(s)")
    for doc in docs:
        print(f"  {doc.metadata['source']}: {len(doc.page_content)} characters")

    if not indexed_splits:
        print(
            "\nThe index is empty. Drop .txt files into ./files and run this again;\n"
            "subfolders are searched, so ./files/_samples works too."
        )
        raise SystemExit(0)

    # With no question given, probe using the corpus's own opening words. That
    # cannot verify the answer, but it does prove the index is reachable and
    # shows which source ranks first.
    queries = sys.argv[1:]
    if not queries:
        queries = [" ".join(docs[0].page_content.split()[:8])]
        print(f"\nNo question given, probing with the opening of {docs[0].metadata['source']!r}.")

    for query in queries:
        print(f"\nQ: {query}")
        results = search_multilingual(query)

        if not results:
            print("  nothing retrieved")
            continue

        for rank, result in enumerate(results, start=1):
            text = result.metadata.get("original_text", result.page_content)
            print(f"  {rank}. {result.metadata['source']}: {' '.join(text.split())[:100]}")
