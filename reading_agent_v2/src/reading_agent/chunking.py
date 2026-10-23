"""Split free text into readable chunks (the agent's actions)."""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?;\n])\s+")


def _n_words(s: str) -> int:
    return len(s.split())


def chunk_text(text, max_words: int = 48, max_chunks: int = 8) -> list[str]:
    """Sentence split, then greedily merge short sentences up to ~max_words.

    Falls back to fixed word windows when there is no punctuation (L7: rule-based
    segmentation fails on unpunctuated / non-whitespace text).
    """
    if text is None or (isinstance(text, float) and text != text):
        return []
    text = str(text).strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    pieces = []
    for s in sents:
        words = s.split()
        if len(words) <= max_words:
            pieces.append(s)
        else:  # over-long sentence -> fixed windows
            pieces += [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    chunks, cur = [], ""
    for p in pieces:
        if cur and _n_words(cur) + _n_words(p) > max_words:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    if len(chunks) > max_chunks:  # merge the tail so no text is dropped silently
        chunks = chunks[:max_chunks - 1] + [" ".join(chunks[max_chunks - 1:])]
    return chunks
