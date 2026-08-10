from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import DEFAULT_PSL_WORDS

_PUNCT_RE = re.compile(r"[\s\u200c]+|[،,.!?؛:()\[\]{}\"'«»]+")


@dataclass(frozen=True, slots=True)
class SignToken:
    word: str
    gloss: str
    asset_hint: str
    known: bool = True


def _latin_slug(index: int) -> str:
    return f"SIGN_{index:03d}"


# The asset names are placeholders by design. Once real recorded clips land in
# assets/signs/, the reverse mode can point to them without touching NLP code.
_PSL_GLOSS_MAP: dict[str, str] = {
    word: f"PSL_{index:03d}_{_latin_slug(index)}" for index, word in enumerate(DEFAULT_PSL_WORDS, start=1)
}
_PSL_GLOSS_MAP.update(
    {
        "مرسی": _PSL_GLOSS_MAP["متشکرم"],
        "تشکر": _PSL_GLOSS_MAP["متشکرم"],
        "آره": _PSL_GLOSS_MAP["بله"],
        "خیر": _PSL_GLOSS_MAP["نه"],
        "کمکم": _PSL_GLOSS_MAP["کمک"],
        "پزشک": _PSL_GLOSS_MAP["دکتر"],
    }
)


def normalize_persian_text(text: str) -> str:
    """Normalize common Arabic/Persian glyph drift. Small but saves headaches."""
    return (
        text.strip()
        .replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ة", "ه")
        .replace("ۀ", "ه")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ؤ", "و")
    )


def tokenize_persian(text: str) -> list[str]:
    normalized = normalize_persian_text(text)
    return [token for token in _PUNCT_RE.split(normalized) if token]


def sentence_to_sign_tokens(text: str) -> list[SignToken]:
    tokens: list[SignToken] = []
    for word in tokenize_persian(text):
        gloss = _PSL_GLOSS_MAP.get(word)
        if gloss is None:
            tokens.append(
                SignToken(
                    word=word,
                    gloss="UNKNOWN",
                    asset_hint="assets/signs/unknown.mp4",
                    known=False,
                )
            )
            continue
        tokens.append(SignToken(word=word, gloss=gloss, asset_hint=f"assets/signs/{gloss}.mp4"))
    return tokens


def known_words() -> list[str]:
    # Return unique canonical words, not aliases.
    return list(DEFAULT_PSL_WORDS)
