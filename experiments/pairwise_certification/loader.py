"""Strict loader for already-exposed, fixed-score development matrices.

This loader never estimates score bounds from the data and never promotes a
previously examined dataset to an independent holdout. The returned matrix is
items by systems, in exactly the source CSV row and system-column order.
"""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_BASENAME = "cs-de_DE_human_scores.csv"
EXPECTED_SHA256 = "f6de1e8da36fb67c7d99ec354e8807a016045c2d5b420f3ab8aeb72816bc279d"
METADATA_COLUMNS = ("sentence_id", "source_text")
RAW_SCORE_BOUNDS = (0.0, 100.0)


def load_development_csv(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load bounded human-score means without filtering or imputing observations.

    A file using the historical Czech--German filename must match its previously
    recorded SHA-256. Only that verified byte content receives the known cost of
    two human ratings per score cell; all other files have unknown rating cost.
    """
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if source.name == EXPECTED_BASENAME and digest != EXPECTED_SHA256:
        raise ValueError("Historical development matrix checksum mismatch")
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8") from exc
    try:
        records = list(csv.reader(io.StringIO(decoded, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError("Malformed CSV") from exc
    if not records:
        raise ValueError("CSV is empty")
    header = records[0]
    if not header or any(not name or name != name.strip() for name in header):
        raise ValueError("Column names must be nonempty without surrounding whitespace")
    if len(set(header)) != len(header):
        raise ValueError("Duplicate column names")
    if any(name not in header for name in METADATA_COLUMNS):
        raise ValueError("CSV requires sentence_id and source_text columns")
    systems = [name for name in header if name not in METADATA_COLUMNS]
    if any(name.casefold() == "refa" for name in systems):
        raise ValueError("Human reference refA must not be included as a system")
    if len(systems) < 2:
        raise ValueError("At least two machine-system score columns are required")
    if len(records) == 1:
        raise ValueError("CSV contains no item rows")
    id_column = header.index("sentence_id")
    score_columns = [header.index(name) for name in systems]
    identifiers: list[str] = []
    rows: list[list[float]] = []
    for line_index, fields in enumerate(records[1:], start=2):
        if len(fields) != len(header):
            raise ValueError(f"Wrong number of fields in CSV record {line_index}")
        item_id = fields[id_column]
        if not item_id or item_id != item_id.strip():
            raise ValueError(f"Empty or whitespace-padded item ID in record {line_index}")
        identifiers.append(item_id)
        try:
            values = [float(fields[column]) for column in score_columns]
        except ValueError as exc:
            raise ValueError(f"Missing or nonnumeric human score in record {line_index}") from exc
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite human score in record {line_index}")
        if not all(RAW_SCORE_BOUNDS[0] <= value <= RAW_SCORE_BOUNDS[1]
                   for value in values):
            raise ValueError(f"Human score outside fixed bounds [0, 100] in record {line_index}")
        rows.append(values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate item ID")
    normalized = np.asarray(rows, dtype=float) / 100.0
    documents = {item_id.rsplit("_#_", 1)[0] for item_id in identifiers}
    metadata: dict[str, Any] = {
        "systems": systems,
        "items": len(rows),
        "documents": len(documents),
        "sha256": digest,
        "score_bounds_raw": list(RAW_SCORE_BOUNDS),
        "human_ratings_per_cell": 2 if digest == EXPECTED_SHA256 else None,
        "data_role": "development_only",
    }
    return normalized, metadata
