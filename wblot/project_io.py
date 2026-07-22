"""Save/load a project (ROIs, labels, settings, results) as JSON."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd


def project_json(state: dict) -> str:
    """Serialize project state to a JSON string (DataFrames become records)."""
    payload = dict(state)
    payload["_saved_at"] = datetime.now().isoformat(timespec="seconds")
    for k, v in list(payload.items()):
        if isinstance(v, pd.DataFrame):
            payload[k] = {"__dataframe__": True, "records": v.to_dict(orient="records")}
    return json.dumps(payload, indent=2, default=str)


def save_project(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(project_json(state))


def load_project(source) -> dict:
    """Load a project from a path, a JSON string/bytes, or a file-like object."""
    if hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, (bytes, bytearray)):
        raw = source
    elif isinstance(source, str) and source.lstrip().startswith("{"):
        raw = source
    else:
        with open(source, "r", encoding="utf-8") as f:
            raw = f.read()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    payload = json.loads(raw)
    for k, v in list(payload.items()):
        if isinstance(v, dict) and v.get("__dataframe__"):
            payload[k] = pd.DataFrame(v["records"])
    return payload
