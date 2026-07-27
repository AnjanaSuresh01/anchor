"""Turn the corpus into author *mentions* — the input to entity resolution.

One row per (paper, author-string) pair. A mention is not a person: "Wei Zhang"
appears three times in this corpus and is almost certainly three different
researchers. Deciding which mentions are the same person is the job of
resolve.py; this module only prepares the evidence.

The features exist because a name on its own cannot disambiguate. What actually
separates two same-named researchers is who they publish with and what they
publish about, so co-authors and subject categories are carried alongside the
name.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from anchor.config import settings

_PUNCT = re.compile(r"[.\-']")
_SPACE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Fold accents so "Müller" and "Muller" compare equal.

    Author strings come from many submission systems and accent handling is
    inconsistent, so this is a normalisation, not a simplification.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def split_name(raw: str) -> tuple[str, str]:
    """Best-effort (first, last) from an arXiv author string.

    Handles both "Yang Zhang" and "Zhang, Yang". Deliberately simple: this
    cannot be right for every naming convention in the world, and pretending
    otherwise would hide the error rather than remove it. Particles like "van"
    and "de" are kept with the surname.
    """
    name = _SPACE.sub(" ", raw.strip())
    if not name:
        return "", ""

    if "," in name:
        last, _, first = name.partition(",")
        return first.strip(), last.strip()

    parts = name.split(" ")
    if len(parts) == 1:
        return "", parts[0]

    particles = {"van", "von", "de", "der", "den", "del", "di", "da", "la", "le", "bin", "al"}
    for i, part in enumerate(parts[:-1]):
        if part.lower() in particles:
            return " ".join(parts[:i]), " ".join(parts[i:])

    return " ".join(parts[:-1]), parts[-1]


def normalise(text: str) -> str:
    return _PUNCT.sub("", strip_accents(text).lower()).strip()


def build_mentions(corpus: Path | None = None) -> list[dict]:
    corpus = corpus or settings.corpus_file
    papers = [
        json.loads(line)
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    mentions: list[dict] = []
    for paper in papers:
        # Full normalised names, not bare surnames. Sharing a co-author called
        # "wang" is close to no evidence — it is the most common surname in the
        # corpus — whereas sharing "xuejie zhang" is strong evidence of one
        # research group.
        full_names = [
            " ".join(filter(None, (normalise(f), normalise(l))))
            for f, l in (split_name(a) for a in paper["authors"])
        ]

        for position, raw in enumerate(paper["authors"]):
            first, last = split_name(raw)
            first_n, last_n = normalise(first), normalise(last)
            self_full = " ".join(filter(None, (first_n, last_n)))

            mentions.append(
                {
                    "mention_id": f"{paper['id']}::{position}",
                    "paper_id": paper["id"],
                    "raw_name": raw,
                    "first_name": first_n,
                    "last_name": last_n,
                    # An initial is all you get from "Y. Zhang", so compare on
                    # it rather than discarding the abbreviated form.
                    "first_initial": first_n[:1],
                    "full_norm": f"{first_n} {last_n}".strip(),
                    # Everyone else on this paper. The strongest available
                    # signal that two same-named mentions are one person.
                    "coauthors": sorted(set(full_names) - {self_full}),
                    "categories": sorted(set(paper["categories"])),
                    "primary_category": paper["primary_category"],
                    "position": position,
                }
            )
    return mentions


def main() -> None:
    mentions = build_mentions()
    out = settings.corpus_file.parent / "mentions.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for m in mentions:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    distinct_raw = len({m["raw_name"] for m in mentions})
    distinct_norm = len({m["full_norm"] for m in mentions})
    print(f"{len(mentions)} mentions from {len({m['paper_id'] for m in mentions})} papers")
    print(f"  {distinct_raw} distinct raw strings")
    print(f"  {distinct_norm} distinct after normalisation "
          f"({distinct_raw - distinct_norm} collapsed by accent/case/punctuation alone)")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
