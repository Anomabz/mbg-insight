"""Progress tracker sederhana berbasis file JSON (tanpa database).

Aman dipakai lokal maupun di hosting (Streamlit Community Cloud):
- Lokal      -> state disimpan permanen di progress.json
- Online     -> filesystem bersifat sementara, jadi state selalu dicerminkan
                ke st.session_state agar tidak hilang saat rerun.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

PROGRESS_PATH = Path(__file__).resolve().parent.parent / "progress.json"
SESSION_KEY = "progress_state"

STATUSES = ["Belum", "Proses", "Selesai"]

DEFAULT_ITEMS = [
    ("load", "Load dataset & sanity check (missing values)"),
    ("eda_uni", "EDA univariat: distribusi target, histogram, statistik deskriptif"),
    ("eda_bi", "EDA bivariat: boxplot, crosstab heatmap, korelasi"),
    ("prep", "Preprocessing & one-hot encoding fitur kategorikal"),
    ("split", "Train-test split stratifikasi 80:20"),
    ("oversample", "Oversampling kelas minoritas pada data latih"),
    ("model", "Training Random Forest"),
    ("eval", "Evaluasi model (classification report & confusion matrix)"),
    ("interpret", "Interpretasi hasil & insight bisnis"),
    ("report", "Laporan akhir / slide presentasi"),
]


def default_state() -> dict:
    return {
        "updated": "",
        "items": [
            {"id": i, "label": label, "status": "Belum", "note": ""}
            for i, label in DEFAULT_ITEMS
        ],
    }


def _read_disk() -> dict:
    if not PROGRESS_PATH.exists():
        return default_state()
    try:
        raw = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_state()

    saved = {item.get("id"): item for item in raw.get("items", [])}
    state = default_state()
    for item in state["items"]:
        old = saved.get(item["id"])
        if old:
            item["status"] = old.get("status", "Belum")
            item["note"] = old.get("note", "")
    state["updated"] = raw.get("updated", "")
    return state


def load_progress() -> dict:
    """Ambil state dari session; kalau belum ada, baca dari disk/default."""
    state = st.session_state.get(SESSION_KEY)
    if state is None:
        state = _read_disk()
        st.session_state[SESSION_KEY] = copy.deepcopy(state)
    return copy.deepcopy(state)


def save_progress(state: dict) -> None:
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state[SESSION_KEY] = copy.deepcopy(state)
    try:
        PROGRESS_PATH.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        # Hosting read-only / ephemeral -> cukup simpan di session.
        pass


def summarize(state: dict) -> tuple[int, int, float]:
    """Kembalikan (selesai, total, persentase selesai)."""
    items = state["items"]
    done = sum(1 for i in items if i["status"] == "Selesai")
    total = len(items)
    return done, total, (done / total * 100 if total else 0.0)
