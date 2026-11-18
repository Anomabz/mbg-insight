"""Pipeline model Random Forest yang dipakai dashboard.

Semua parameter penting dikumpulkan di satu tempat agar mudah dijelaskan
saat presentasi: 5 fitur fokus, target, dan 2 kelas biner rekomendasi dosen.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

TARGET = "primary_food_waste_driver"

CAT_FEATURES = ["beneficiary_category", "protein_type"]
NUM_FEATURES = [
    "food_arrival_temperature_celsius",
    "transit_time_minutes",
    "plate_waste_percentage",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

BINARY_CLASSES = ["Prior-Snack-Consumption", "Excessive-Portion-Size"]
MAJORITY_CLASS = "Prior-Snack-Consumption"
MINORITY_CLASS = "Excessive-Portion-Size"

MODES = {"biner": "Biner (2 kelas)", "multikelas": "Multikelas (5 kelas)"}


def subset_for_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter data sesuai mode pemodelan."""
    if mode == "biner":
        return df[df[TARGET].isin(BINARY_CLASSES)].copy()
    return df.copy()


def build_pipeline(n_estimators: int, max_depth: int | None, random_state: int) -> Pipeline:
    """ColumnTransformer + RandomForest dalam satu Pipeline."""
    prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_FEATURES),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("prep", prep),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


@st.cache_resource(show_spinner="Melatih model Random Forest...")
def train_model(
    _df: pd.DataFrame,
    mode: str,
    n_estimators: int,
    max_depth: int | None,
    test_size: float,
    random_state: int,
    n_rows: int,
    data_sig: tuple = (),
) -> dict:
    """Latih model dan kembalikan semua artefak evaluasi.

    Parameter ``_df`` diawali underscore supaya Streamlit tidak meng-hash
    seluruh DataFrame (17 MB) setiap kali tombol ditekan; ``n_rows`` dipakai
    sebagai penanda cache supaya model dilatih ulang saat data berubah.
    """
    data = subset_for_mode(_df, mode)
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Oversampling hanya untuk kelas biner (kelas minoritas digandakan).
    oversampled = 0
    if mode == "biner":
        train_df = pd.concat([X_train, y_train], axis=1)
        maj = train_df[train_df[TARGET] == MAJORITY_CLASS]
        minor = train_df[train_df[TARGET] == MINORITY_CLASS]
        if len(minor) < len(maj):
            minor_up = resample(
                minor, replace=True, n_samples=len(maj), random_state=random_state
            )
            train_df = pd.concat([maj, minor_up])
            oversampled = len(minor_up) - len(minor)
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]

    model = build_pipeline(n_estimators, max_depth, random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = list(model.classes_)

    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T
    report = report.loc[[l for l in labels if l in report.index]]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    prep = model.named_steps["prep"]
    importance = pd.DataFrame(
        {
            "fitur": prep.get_feature_names_out(),
            "importance": model.named_steps["rf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "mode": mode,
        "model": model,
        "labels": labels,
        "report": report,
        "confusion": cm,
        "importance": importance,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_oversampled": oversampled,
        "class_balance": y_train.value_counts(),
    }


def predict_one(model: Pipeline, values: dict) -> tuple[str, pd.DataFrame]:
    """Prediksi satu paket dari input form; kembalikan label + probabilitas."""
    row = pd.DataFrame([{f: values[f] for f in FEATURES}])
    label = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    table = (
        pd.DataFrame({"kelas": model.classes_, "probabilitas": proba})
        .sort_values("probabilitas", ascending=False)
        .reset_index(drop=True)
    )
    return str(label), table
