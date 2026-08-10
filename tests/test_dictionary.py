from __future__ import annotations

import unittest

from psl_translator.dictionary import normalize_persian_text, sentence_to_sign_tokens, tokenize_persian


class DictionaryTests(unittest.TestCase):
    def test_normalizes_arabic_letters(self) -> None:
        self.assertEqual(normalize_persian_text("سلام يكي"), "سلام یکی")

    def test_tokenize_persian(self) -> None:
        self.assertEqual(tokenize_persian("سلام، لطفا آب!"), ["سلام", "لطفا", "آب"])

    def test_sentence_to_sign_marks_unknown(self) -> None:
        tokens = sentence_to_sign_tokens("سلام قورمهسبزی")
        self.assertTrue(tokens[0].known)
        self.assertFalse(tokens[1].known)
        self.assertEqual(tokens[1].gloss, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
