"""Task 1.2 - label mapping.

The raw CIC-IDS2017 CSVs carry 14 attack labels plus BENIGN. Neural Cleanse is
mathematically inert on a binary task (Phase 0, G-02), so the project runs
multi-class: 14 raw labels collapse to 8 families.

The ``\\x96`` in the Web Attack labels is a Windows-1252 en-dash, not an ASCII
hyphen - matching on ``'Web Attack - Brute Force'`` silently yields zero rows.
"""

from __future__ import annotations

CLASS_NAMES = [
    "Benign", "DoS", "DDoS", "PortScan",
    "BruteForce", "WebAttack", "Bot", "Infiltration",
]
N_CLASSES = len(CLASS_NAMES)

# raw label (whitespace-stripped) -> family id
LABEL_MAP = {
    "BENIGN": 0,

    "DoS Hulk": 1, "DoS GoldenEye": 1, "DoS slowloris": 1,
    "DoS Slowhttptest": 1, "Heartbleed": 1,

    "DDoS": 2,
    "PortScan": 3,

    "FTP-Patator": 4, "SSH-Patator": 4,

    "Web Attack \x96 Brute Force": 5,
    "Web Attack \x96 XSS": 5,
    "Web Attack \x96 Sql Injection": 5,
    # tolerate the ASCII-hyphen variant some corrected releases ship
    "Web Attack - Brute Force": 5,
    "Web Attack - XSS": 5,
    "Web Attack - Sql Injection": 5,

    "Bot": 6,
    "Infiltration": 7,
}

# The separator in the Web Attack labels is the one thing that varies between
# CIC-IDS2017 redistributions: the original CSVs carry a Windows-1252 en-dash
# (code point 0x96), mirrors transcoded to UTF-8 with errors='replace' carry
# U+FFFD instead, and some corrected releases ship a plain ASCII hyphen.
# Matching the literal bytes makes every new mirror a fresh unmapped label, so
# every variant is folded to '-' before the lookup. Spelled as code points so
# this module stays pure ASCII and survives any re-encoding of the file.
_SEPARATORS = tuple(chr(c) for c in (0x96, 0xFFFD, 0x2013, 0x2014))


def _canonical(label: str) -> str:
    """Separator- and whitespace-normalised form used for LABEL_MAP lookups."""
    for ch in _SEPARATORS:
        label = label.replace(ch, "-")
    return " ".join(label.split())


# LABEL_MAP above stays as the record of spellings actually observed in the
# wild; lookups go through the normalised view of it.
_CANONICAL_MAP = {_canonical(k): v for k, v in LABEL_MAP.items()}

BINARY_MAP = None  # sentinel: "anything not BENIGN -> 1" (kept as an ablation flag)

# Rare-class decision (Task 1.2 verification):
#   Heartbleed (~11 rows) is folded into DoS (family 1) - it is a DoS-class
#   OpenSSL exploit and cannot support its own class.
#   Infiltration (~36 rows) is kept as its own family 7 but flagged rare; any
#   per-class metric on it is reported with an explicit low-support caveat and
#   it is excluded from macro-F1 headline numbers when support < 50 in a split.
RARE_CLASSES = {7}
RARE_SUPPORT_THRESHOLD = 50


def map_labels(raw_labels, binary: bool = False):
    """Map an iterable of raw string labels to integer class ids.

    Raises on any unmapped label - never silently drops (Task 1.2 verification).
    """
    import numpy as np

    out = np.empty(len(raw_labels), dtype=np.int64)
    unknown = set()
    for i, lab in enumerate(raw_labels):
        lab = str(lab).strip()
        if binary:
            out[i] = 0 if lab.upper() == "BENIGN" else 1
            continue
        key = _canonical(lab)
        if key not in _CANONICAL_MAP:
            unknown.add(lab)
            continue
        out[i] = _CANONICAL_MAP[key]
    if unknown:
        raise ValueError(f"unmapped labels (fix flids/data/labels.py): {sorted(unknown)}")
    return out


def class_counts(y):
    import numpy as np
    counts = np.bincount(np.asarray(y), minlength=N_CLASSES)
    return {CLASS_NAMES[i]: int(counts[i]) for i in range(N_CLASSES)}
