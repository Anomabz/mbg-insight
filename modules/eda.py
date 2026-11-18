"""Kumpulan grafik Plotly untuk tab eksplorasi data."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"

# Warna konsisten untuk 5 kelas pemicu food waste
DRIVER_COLORS = {
    "Prior-Snack-Consumption": "#4C78A8",
    "Excessive-Portion-Size": "#F58518",
    "Sub-Optimal-Arrival-Temperature": "#E45756",
    "Delivery-Transit-Delay": "#72B7B2",
    "Low-Palatability-Menu": "#54A24B",
}


def _style(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def target_distribution(df: pd.DataFrame, target: str) -> go.Figure:
    """Bar chart jumlah & persentase tiap kelas target (univariat)."""
    vc = df[target].value_counts().reset_index()
    vc.columns = [target, "jumlah"]
    vc["persen"] = vc["jumlah"] / vc["jumlah"].sum() * 100

    fig = px.bar(
        vc.sort_values("jumlah"),
        x="jumlah",
        y=target,
        orientation="h",
        color=target,
        color_discrete_map=DRIVER_COLORS,
        text=vc.sort_values("jumlah")["persen"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Jumlah paket", yaxis_title="")
    return _style(fig, "Distribusi Kelas Pemicu Food Waste")


def category_counts(df: pd.DataFrame, col: str) -> go.Figure:
    """Bar chart frekuensi kolom kategorikal (univariat)."""
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "jumlah"]
    fig = px.bar(
        vc.sort_values("jumlah"),
        x="jumlah",
        y=col,
        orientation="h",
        text="jumlah",
        color="jumlah",
        color_continuous_scale="Teal",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      xaxis_title="Jumlah sampel", yaxis_title="")
    return _style(fig, f"Frekuensi {col}", height=320)


def numeric_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogram + box marginal untuk fitur numerik (univariat)."""
    fig = px.histogram(df, x=col, nbins=40, marginal="box", color_discrete_sequence=["#4C78A8"])
    fig.update_layout(bargap=0.05, yaxis_title="Jumlah", xaxis_title=col)
    return _style(fig, f"Distribusi {col}", height=340)


def numeric_by_target(df: pd.DataFrame, col: str, target: str) -> go.Figure:
    """Boxplot fitur numerik per kelas target (bivariat)."""
    fig = px.box(df, x=target, y=col, color=target, color_discrete_map=DRIVER_COLORS, points="outliers")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=col)
    fig.update_xaxes(tickangle=-25)
    return _style(fig, f"{col} per Pemicu", height=380)


def crosstab_heatmap(df: pd.DataFrame, cat: str, target: str) -> go.Figure:
    """Heatmap proporsi kelas target di setiap kategori (bivariat)."""
    ct = pd.crosstab(df[cat], df[target], normalize="index") * 100
    fig = px.imshow(
        ct,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="% dalam kategori"),
    )
    fig.update_layout(coloraxis_colorbar=dict(title="%"), xaxis_title="", yaxis_title="")
    fig.update_xaxes(tickangle=-25)
    return _style(fig, f"{cat} vs Pemicu (%)", height=380)


def correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Matriks korelasi Pearson antar fitur numerik (bivariat)."""
    corr = df[cols].corr().round(3)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    return _style(fig, "Korelasi Fitur Numerik", height=420)


def scatter_pair(df: pd.DataFrame, x: str, y: str, target: str) -> go.Figure:
    """Scatter dua fitur numerik diwarnai kelas target (bivariat)."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=target,
        color_discrete_map=DRIVER_COLORS,
        opacity=0.6,
    )
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return _style(fig, f"{y} vs {x}", height=420)


def province_plate_waste(df: pd.DataFrame) -> go.Figure:
    """Rata-rata plate waste per provinsi."""
    agg = (
        df.groupby("province")["plate_waste_percentage"]
        .mean()
        .reset_index()
        .sort_values("plate_waste_percentage")
    )
    fig = px.bar(
        agg,
        x="plate_waste_percentage",
        y="province",
        orientation="h",
        text=agg["plate_waste_percentage"].map(lambda v: f"{v:.1f}%"),
        color="plate_waste_percentage",
        color_continuous_scale="OrRd",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_title="Rata-rata plate waste (%)", yaxis_title="")
    return _style(fig, "Rata-rata Plate Waste per Provinsi", height=420)


def compliance_distribution(df: pd.DataFrame) -> go.Figure:
    vc = df["bgn_program_compliance_tier"].value_counts().reset_index()
    vc.columns = ["tier", "jumlah"]
    fig = px.pie(vc, names="tier", values="jumlah", hole=0.55,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textinfo="percent+label")
    return _style(fig, "Tingkat Kepatuhan Standar BGN", height=360)
