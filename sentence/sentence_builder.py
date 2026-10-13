"""Collects accepted gestures and manages the sentence being built.

Two layers:
* tokens          - completed words (I, NEED, WATER, ALI ...)
* pending buffer  - letters/digits still being spelled; SPACE completes them
"""
from __future__ import annotations

from typing import Callable

from sentence.phrase_engine import PhraseEngine, Token


class SentenceBuilder:
    def __init__(self, engine: PhraseEngine | None = None,
                 name_getter: Callable[[], str] | None = None):
        self.engine = engine or PhraseEngine()
        self.name_getter = name_getter or (lambda: "")
        self.tokens: list[Token] = []
        self.pending = ""             # letters or digits being spelled
        self.pending_kind = "letters"  # "letters" | "number"

    # ------------------------------------------------------------ adding input
    def add_letter(self, letter: str) -> None:
        letter = letter.strip().upper()
        if not letter.isalpha():
            return
        if self.pending and self.pending_kind != "letters":
            self.commit_pending()
        self.pending_kind = "letters"
        self.pending += letter

    def add_digit(self, digit: str) -> None:
        digit = digit.strip()
        if not digit.isdigit():
            return
        if self.pending and self.pending_kind != "number":
            self.commit_pending()
        self.pending_kind = "number"
        self.pending += digit

    def add_word(self, word: str, kind: str = "sign") -> None:
        word = " ".join(word.strip().split())
        if not word:
            return
        self.commit_pending()
        self.tokens.append(Token(word.upper(), kind))

    def add_sentence(self, text: str) -> None:
        """A whole ready-made sentence (from a SENTENCES-mode gesture). {name} is filled in later."""
        text = " ".join(text.strip().split())
        if not text:
            return
        self.commit_pending()
        self.tokens.append(Token(text, "sentence"))

    def add_gesture(self, label: str, category: str, output: str = "") -> str:
        """Route an accepted gesture to the right place. Returns what happened:
        'letter', 'digit', 'word' or 'sentence'."""
        if category == "sentence":
            self.add_sentence(output or label)
            return "sentence"
        label = label.strip().upper()
        if category == "alphabet" and len(label) == 1:
            self.add_letter(label)
            return "letter"
        if category == "number" and label.isdigit():
            self.add_digit(label)
            return "digit"
        self.add_word(label)
        return "word"

    def space(self) -> str:
        """Finish the letters being spelled. Returns the finished word (or '')."""
        return self.commit_pending()

    def commit_pending(self) -> str:
        if not self.pending:
            return ""
        word = self.pending
        kind = self.pending_kind
        self.tokens.append(Token(word, "letters" if kind == "letters" else "number"))
        self.pending = ""
        return word

    # ---------------------------------------------------------------- editing
    def delete(self) -> bool:
        """Delete the last letter being spelled, otherwise the last word."""
        if self.pending:
            self.pending = self.pending[:-1]
            return True
        if self.tokens:
            self.tokens.pop()
            return True
        return False

    def clear(self) -> None:
        self.tokens = []
        self.pending = ""

    def capitalize_last(self) -> bool:
        """Toggle a forced capital letter on the last word."""
        self.commit_pending()
        if not self.tokens:
            return False
        self.tokens[-1].force_cap = not self.tokens[-1].force_cap
        return True

    # ------------------------------------------------------------- name helpers
    def use_as_name(self) -> str:
        """Turn the last spelled word into 'My name is <Name>'. Returns the name."""
        self.commit_pending()
        for index in range(len(self.tokens) - 1, -1, -1):
            if self.tokens[index].kind == "letters":
                name = self.tokens[index].text.title()
                self.tokens[index:index + 1] = [
                    Token("MY"), Token("NAME"), Token("IS"), Token(name.upper(), "name")]
                return name
        return ""

    def insert_name_phrase(self, name: str) -> bool:
        """Append 'My name is <name>' using a configured name."""
        name = name.strip()
        if not name:
            return False
        self.commit_pending()
        self.tokens += [Token("MY"), Token("NAME"), Token("IS"), Token(name.upper(), "name")]
        return True

    def load_phrase(self, phrase: str) -> None:
        """Replace the sentence with the words of a ready-made phrase."""
        self.clear()
        cleaned = "".join(ch if ch.isalnum() or ch in " '" else " " for ch in phrase)
        for word in cleaned.split():
            self.tokens.append(Token(word.upper(), "typed"))

    # ------------------------------------------------------------------ output
    def _all_tokens(self) -> list[Token]:
        tokens = list(self.tokens)
        if self.pending:
            tokens.append(Token(self.pending, "letters" if self.pending_kind == "letters" else "number"))
        return tokens

    def sentence(self) -> str:
        """The nicely formatted sentence, e.g. 'I need water.'"""
        return self.engine.render(self._all_tokens(), self.name_getter())

    def raw_text(self) -> str:
        """The plain detected words, e.g. 'I NEED WATER'."""
        name = self.name_getter().strip().title() or "…"
        return " ".join(f"[{t.text.replace('{name}', name)}]" if t.kind == "sentence" else t.text
                        for t in self._all_tokens())

    @property
    def is_empty(self) -> bool:
        return not self.tokens and not self.pending
