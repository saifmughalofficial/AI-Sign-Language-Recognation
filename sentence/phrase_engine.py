"""Turns a list of word tokens into a natural sentence.

Rules come from data/phrases.json so they can be changed without touching code:

* templates   - exact patterns, e.g. WHERE + {w}  ->  "Where is the {w}?"
                ("alone": true = only when it is the whole sentence, e.g. HELP -> "Help!")
* copula      - "I" + HUNGRY  ->  "I am hungry"
* subject-less verbs - NEED + WATER -> "I need water"
* question words - sentences that start with WHERE/WHAT/... end with "?"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PROPER_KINDS = {"letters", "name"}       # spelled words / names are capitalised


@dataclass
class Token:
    text: str
    kind: str = "sign"        # sign | letters | name | number | typed | sentence
    force_cap: bool = False   # set by the CAPITALIZE button


@dataclass
class _Item:
    text: str                 # display text
    word: str                 # UPPERCASE original (used for rules)
    final: bool = False       # already formatted (from a template)
    proper: bool = False
    force_cap: bool = False


class PhraseEngine:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.templates = sorted(
            [t for t in cfg.get("templates", [])
             if isinstance(t, dict) and t.get("match") and "output" in t],
            key=lambda t: -len(t["match"]))
        self.function_words = {w.upper() for w in cfg.get("function_words", [])}
        self.copula_words = {w.upper() for w in cfg.get("copula_words", [])}
        self.subject_less_verbs = {w.upper() for w in cfg.get("subject_less_verbs", [])}
        self.question_words = {w.upper() for w in cfg.get("question_words", [])}

    # ------------------------------------------------------------------ public
    def render(self, tokens: Iterable[Token], name: str = "") -> str:
        toks = list(tokens)
        if not toks:
            return ""
        items = self._apply_templates(toks, name.strip())
        items = self._apply_rules(items)
        return self._join(items)

    # ---------------------------------------------------------------- templates
    @staticmethod
    def _is_placeholder(pattern: str) -> bool:
        return len(pattern) > 2 and pattern.startswith("{") and pattern.endswith("}")

    def _matches(self, template: dict, toks: list[Token], start: int) -> dict | None:
        pattern = template["match"]
        if start + len(pattern) > len(toks):
            return None
        if template.get("alone") and not (start == 0 and len(toks) == len(pattern)):
            return None            # "alone" templates only apply when they are the whole sentence
        captured: dict[str, Token] = {}
        for offset, pat in enumerate(pattern):
            tok = toks[start + offset]
            up = tok.text.strip().upper()
            if self._is_placeholder(pat):
                if up in self.function_words:      # "IS", "THE"... never fill a slot
                    return None
                captured[pat[1:-1]] = tok
            elif up != pat.upper():
                return None
        return captured

    def _fill(self, output: str, captured: dict[str, Token], name: str) -> str:
        text = output
        for key, tok in captured.items():
            value = tok.text.strip().title() if key == "n" else tok.text.strip().lower()
            text = text.replace("{" + key + "}", value)
        text = text.replace("{name}", name.strip().title())
        return " ".join(text.split())          # tidy double spaces when {name} is empty

    def _apply_templates(self, toks: list[Token], name: str) -> list[_Item]:
        items: list[_Item] = []
        i = 0
        while i < len(toks):
            for template in self.templates:
                captured = self._matches(template, toks, i)
                if captured is not None:
                    text = self._fill(template["output"], captured, name)
                    items.append(_Item(text, text.upper(), final=True,
                                       force_cap=toks[i].force_cap))
                    i += len(template["match"])
                    break
            else:
                tok = toks[i]
                if tok.kind == "sentence":        # ready-made sentence: keep as written
                    text = self._fill(tok.text, {}, name or "…")
                    items.append(_Item(text, text.upper(), final=True))
                    i += 1
                    continue
                up = tok.text.strip().upper()
                items.append(_Item(tok.text.strip(), up,
                                   proper=tok.kind in PROPER_KINDS,
                                   force_cap=tok.force_cap))
                i += 1
        return items

    # -------------------------------------------------------------------- rules
    def _apply_rules(self, items: list[_Item]) -> list[_Item]:
        if not items:
            return items
        # "NEED WATER" -> "I need water"
        if not items[0].final and items[0].word in self.subject_less_verbs:
            items.insert(0, _Item("I", "I"))
        # "I HUNGRY" -> "I am hungry"
        out: list[_Item] = []
        for idx, item in enumerate(items):
            out.append(item)
            nxt = items[idx + 1] if idx + 1 < len(items) else None
            if nxt and not item.final and not nxt.final and nxt.word in self.copula_words:
                verb = {"I": "am", "YOU": "are", "WE": "are", "THEY": "are",
                        "HE": "is", "SHE": "is", "IT": "is"}.get(item.word)
                if verb:
                    out.append(_Item(verb, verb.upper()))
        return out

    # --------------------------------------------------------------------- join
    def _join(self, items: list[_Item]) -> str:
        parts: list[str] = []
        capitalize_next = True
        for item in items:
            if item.final:
                text = item.text
            elif item.word == "I":
                text = "I"
            elif item.proper:
                text = item.text.title()
            elif item.text.isdigit():
                text = item.text
            else:
                text = item.text.lower()
            if (capitalize_next or item.force_cap) and text:
                text = text[0].upper() + text[1:]
            parts.append(text)
            capitalize_next = text.endswith((".", "!", "?"))
        sentence = " ".join(parts).strip()
        if not sentence:
            return ""
        if not sentence.endswith((".", "!", "?")):
            first = items[0].word if not items[0].final else ""
            sentence += "?" if first in self.question_words else "."
        return sentence
