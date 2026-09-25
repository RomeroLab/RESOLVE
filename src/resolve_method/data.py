from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_SEQ = ("sequence", "seq", "mutated_sequence", "aa_sequence")
_LABEL = ("label", "score", "fitness", "target", "dms_score")
_ELITE = ("elite", "is_elite")
_AA = set("ACDEFGHIKLMNPQRSTVWY")

def _pick(fieldnames: list[str], options: tuple[str, ...]) -> str | None:
    folded = {name.casefold().strip(): name for name in fieldnames}
    for option in options:
        if option in folded:
            return folded[option]
    return None

def load_csv(path: str | Path) -> dict:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        seq_col = _pick(list(reader.fieldnames), _SEQ)
        label_col = _pick(list(reader.fieldnames), _LABEL)
        elite_col = _pick(list(reader.fieldnames), _ELITE)
        if seq_col is None or label_col is None:
            raise ValueError(
                f"{path} needs a sequence column ({', '.join(_SEQ)}) and a "
                f"label column ({', '.join(_LABEL)}). Found {list(reader.fieldnames)}"
            )
        sequences, labels, elites = [], [], []
        for row_number, row in enumerate(reader, start=2):
            seq = (row.get(seq_col) or "").strip().upper()
            if not seq or any(ch not in _AA for ch in seq):
                raise ValueError(f"{path}:{row_number} is not a protein sequence of the 20 amino acids")
            try:
                label = float(row[label_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number} label is not a number") from exc
            if not np.isfinite(label):
                raise ValueError(f"{path}:{row_number} label is not finite")
            sequences.append(seq)
            labels.append(label)
            if elite_col is not None:
                elites.append(str(row.get(elite_col, "")).strip().lower() in {"1", "true", "yes"})
    if not sequences:
        raise ValueError(f"{path} has no rows")
    out = {
        "sequences": sequences,
        "y": np.asarray(labels, dtype=np.float64),
        "elite": np.asarray(elites, dtype=bool) if elite_col else None,
    }
    return out
