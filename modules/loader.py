"""Pemuatan dataset MBG dengan cache Streamlit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_FILE = "indonesia_mbg_nationwide_enriched_2026.csv"
BASE_DIR = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner="Memuat dataset MBG...")
def load_data() -> pd.DataFrame:
    """Muat seluruh dataset (sekali saja, lalu di-cache)."""
    path = BASE_DIR / DATA_FILE
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if str(df[col].dtype) in ("object", "str", "string"):
            df[col] = df[col].str.strip()
    return df


def apply_filters(
    df: pd.DataFrame,
    provinces: list[str],
    clusters: list[str],
    beneficiaries: list[str],
    proteins: list[str],
    tiers: list[str],
) -> pd.DataFrame:
    """Terapkan filter sidebar. List kosong berarti 'semua'."""
    out = df
    if provinces:
        out = out[out["province"].isin(provinces)]
    if clusters:
        out = out[out["regional_cluster"].isin(clusters)]
    if beneficiaries:
        out = out[out["beneficiary_category"].isin(beneficiaries)]
    if proteins:
        out = out[out["protein_type"].isin(proteins)]
    if tiers:
        out = out[out["bgn_program_compliance_tier"].isin(tiers)]
    return out


def fmt_int(value: float) -> str:
    """Format angka gaya Indonesia: 13.000"""
    return f"{int(value):,}".replace(",", ".")
