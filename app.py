"""Dashboard MBG Insight — eksplorasi data, performa model, prediksi, dan progres.

Jalankan dengan:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import eda, loader, modeling, tracker

st.set_page_config(
    page_title="MBG Insight",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DRIVER_SHORT = {
    "Prior-Snack-Consumption": "Jajan sebelum makan",
    "Excessive-Portion-Size": "Porsi berlebih",
    "Sub-Optimal-Arrival-Temperature": "Suhu tiba kurang optimal",
    "Delivery-Transit-Delay": "Keterlambatan pengiriman",
    "Low-Palatability-Menu": "Menu kurang disukai",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df_all = loader.load_data()


def data_signature(frame: pd.DataFrame) -> tuple:
    """Penanda cache model: berubah kalau isi data berubah."""
    counts = tuple(sorted(frame[modeling.TARGET].value_counts().items()))
    return (len(frame), counts)


# ----------------------------------------------------------------------------
# Sidebar — filter global
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🍱 MBG Insight")
    st.caption("Dashboard Program Makan Bergizi Gratis")

    def _multi(label: str, col: str) -> list[str]:
        options = sorted(df_all[col].dropna().unique().tolist())
        return st.multiselect(label, options, default=[], placeholder="Semua")

    provinces = _multi("Provinsi", "province")
    clusters = _multi("Klaster regional", "regional_cluster")
    beneficiaries = _multi("Kategori penerima", "beneficiary_category")
    proteins = _multi("Jenis protein", "protein_type")
    tiers = _multi("Tingkat kepatuhan BGN", "bgn_program_compliance_tier")

    st.divider()
    st.caption(f"Total dataset: **{loader.fmt_int(len(df_all))}** paket")

df = loader.apply_filters(
    df_all, provinces, clusters, beneficiaries, proteins, tiers
)

if df.empty:
    st.warning("Tidak ada data yang cocok dengan filter. Silakan kurangi filter di sidebar.")
    st.stop()

st.caption(
    f"Menampilkan **{loader.fmt_int(len(df))}** dari {loader.fmt_int(len(df_all))} paket "
    f"({len(df) / len(df_all) * 100:.1f}%)"
)

tab_overview, tab_eda, tab_model, tab_prediksi, tab_data, tab_progress = st.tabs(
    [
        "📊 Overview",
        "🔍 Eksplorasi Data",
        "🤖 Performa Model",
        "🔮 Prediksi",
        "📋 Data Explorer",
        "✅ Progress",
    ]
)


# ----------------------------------------------------------------------------
# 1. Overview
# ----------------------------------------------------------------------------
with tab_overview:
    avg_waste = df["plate_waste_percentage"].mean()
    avg_temp = df["food_arrival_temperature_celsius"].mean()
    compliant = df["bgn_program_compliance_tier"].isin(
        ["Standard-Compliant", "Exemplary-Gold-Standard"]
    ).mean() * 100
    top_driver = df[modeling.TARGET].value_counts().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Paket", loader.fmt_int(len(df)))
    c2.metric("Rata-rata Plate Waste", f"{avg_waste:.1f}%")
    c3.metric("Rata-rata Suhu Tiba", f"{avg_temp:.1f} °C")
    c4.metric("Sesuai Standar BGN", f"{compliant:.1f}%")

    st.info(
        f"Pemicu food waste terbanyak pada seleksi data ini: **{top_driver}** "
        f"({DRIVER_SHORT.get(top_driver, '-')})."
    )

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(eda.target_distribution(df, modeling.TARGET), width="stretch")
    with right:
        st.plotly_chart(eda.compliance_distribution(df), width="stretch")

    st.plotly_chart(eda.province_plate_waste(df), width="stretch")


# ----------------------------------------------------------------------------
# 2. Eksplorasi Data
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Univariat — melihat satu variabel")
    cols = st.columns(3)
    for col, name in zip(cols, modeling.NUM_FEATURES):
        with col:
            st.plotly_chart(eda.numeric_histogram(df, name), width="stretch")

    st.subheader("Bivariat — melihat hubungan dua variabel")
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(
            eda.category_counts(df, "beneficiary_category"), width="stretch"
        )
    with cols[1]:
        st.plotly_chart(eda.category_counts(df, "protein_type"), width="stretch")

    picked_num = st.selectbox("Pilih fitur numerik", modeling.NUM_FEATURES, index=2)
    st.plotly_chart(
        eda.numeric_by_target(df, picked_num, modeling.TARGET), width="stretch"
    )

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(
            eda.crosstab_heatmap(df, "beneficiary_category", modeling.TARGET),
            width="stretch",
        )
    with cols[1]:
        st.plotly_chart(
            eda.crosstab_heatmap(df, "protein_type", modeling.TARGET),
            width="stretch",
        )

    st.plotly_chart(
        eda.correlation_heatmap(df, modeling.NUM_FEATURES), width="stretch"
    )

    x_col, y_col = st.columns(2)
    x_feat = x_col.selectbox("Sumbu X", modeling.NUM_FEATURES, index=0)
    y_feat = y_col.selectbox("Sumbu Y", modeling.NUM_FEATURES, index=2)
    st.plotly_chart(
        eda.scatter_pair(df, x_feat, y_feat, modeling.TARGET), width="stretch"
    )


# ----------------------------------------------------------------------------
# 3. Performa Model
# ----------------------------------------------------------------------------
with tab_model:
    st.subheader("Latih & evaluasi Random Forest")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        mode_label = c1.radio(
            "Skenario", list(modeling.MODES.values()), index=0, horizontal=False
        )
        mode = [k for k, v in modeling.MODES.items() if v == mode_label][0]
        n_estimators = c2.slider("Jumlah pohon", 50, 500, 200, step=50)
        max_depth = c3.slider("Kedalaman maksimum", 2, 30, 12, step=1)
        test_size = c4.slider("Proporsi data uji", 0.1, 0.4, 0.2, step=0.05)

    if len(df) < 50:
        st.warning("Data terlalu sedikit untuk melatih model (minimal 50 baris).")
    else:
        result = modeling.train_model(
            df,
            mode,
            n_estimators,
            max_depth,
            test_size,
            42,
            len(df),
            data_signature(df),
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Akurasi", f"{result['accuracy'] * 100:.2f}%")
        m2.metric("Balanced accuracy", f"{result['balanced_accuracy'] * 100:.2f}%")
        m3.metric("Data latih", loader.fmt_int(result["n_train"]))
        m4.metric("Data uji", loader.fmt_int(result["n_test"]))

        if result["mode"] == "biner":
            st.caption(
                f"Oversampling: {loader.fmt_int(result['n_oversampled'])} baris minoritas "
                f"digandakan pada data latih saja."
            )

        if result["accuracy"] > 0.99:
            st.warning(
                "Akurasi mendekati 100%. Dataset ini sintetis dan aturan penentuan labelnya "
                "deterministik (lihat kolom `cot_nutrition_logistics_rationale`), sehingga "
                "hasil ini bukan bukti model akan sehebat itu pada data dunia nyata. "
                "Siapkan penjelasan ini saat presentasi."
            )

        left, right = st.columns([1, 1])
        with left:
            cm = result["confusion"]
            labels = result["labels"]
            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(color="Jumlah"),
            )
            fig_cm.update_layout(
                template=eda.TEMPLATE,
                height=420,
                title=dict(text="Confusion Matrix (data uji)", font_size=14),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            fig_cm.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_cm, width="stretch")

        with right:
            imp = result["importance"].head(10).iloc[::-1]
            fig_imp = px.bar(
                imp,
                x="importance",
                y="fitur",
                orientation="h",
                text=imp["importance"].map(lambda v: f"{v:.3f}"),
                color="importance",
                color_continuous_scale="Teal",
            )
            fig_imp.update_traces(textposition="outside")
            fig_imp.update_layout(
                template=eda.TEMPLATE,
                height=420,
                showlegend=False,
                coloraxis_showscale=False,
                title=dict(text="10 Fitur Paling Berpengaruh", font_size=14),
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis_title="Importance",
                yaxis_title="",
            )
            st.plotly_chart(fig_imp, width="stretch")

        st.markdown("**Classification report (data uji)**")
        st.dataframe(
            result["report"].style.format("{:.3f}"),
            width="stretch",
        )


# ----------------------------------------------------------------------------
# 4. Prediksi
# ----------------------------------------------------------------------------
with tab_prediksi:
    st.subheader("Prediksi pemicu food waste untuk satu paket")
    st.caption(
        "Model dilatih ulang secara otomatis pada data terfilter memakai skenario biner "
        "(Prior-Snack-Consumption vs Excessive-Portion-Size)."
    )

    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        ben = c1.selectbox("Kategori penerima", sorted(df["beneficiary_category"].unique()))
        prot = c2.selectbox("Jenis protein", sorted(df["protein_type"].unique()))

        c3, c4, c5 = st.columns(3)
        temp = c3.slider(
            "Suhu tiba (°C)",
            float(df["food_arrival_temperature_celsius"].min()),
            float(df["food_arrival_temperature_celsius"].max()),
            float(df["food_arrival_temperature_celsius"].median()),
            step=0.5,
        )
        transit = c4.slider(
            "Waktu transit (menit)",
            int(df["transit_time_minutes"].min()),
            int(df["transit_time_minutes"].max()),
            int(df["transit_time_minutes"].median()),
        )
        waste = c5.slider(
            "Plate waste (%)",
            float(df["plate_waste_percentage"].min()),
            float(df["plate_waste_percentage"].max()),
            float(df["plate_waste_percentage"].median()),
            step=0.5,
        )

        submitted = st.form_submit_button("Prediksi", type="primary")

    if submitted:
        binary_df = modeling.subset_for_mode(df, "biner")
        if len(binary_df) < 50:
            st.error("Data kelas biner tidak cukup untuk melatih model.")
        else:
            model_result = modeling.train_model(
                df, "biner", 200, 12, 0.2, 42, len(df), data_signature(df)
            )
            label, table = modeling.predict_one(
                model_result["model"],
                {
                    "beneficiary_category": ben,
                    "protein_type": prot,
                    "food_arrival_temperature_celsius": temp,
                    "transit_time_minutes": transit,
                    "plate_waste_percentage": waste,
                },
            )

            st.success(
                f"Prediksi: **{label}** — {DRIVER_SHORT.get(label, '')} "
                f"({table.loc[0, 'probabilitas'] * 100:.1f}% keyakinan)"
            )

            fig_p = px.bar(
                table.iloc[::-1],
                x="probabilitas",
                y="kelas",
                orientation="h",
                text=table.iloc[::-1]["probabilitas"].map(lambda v: f"{v * 100:.1f}%"),
                color="probabilitas",
                color_continuous_scale="Blues",
            )
            fig_p.update_traces(textposition="outside")
            fig_p.update_layout(
                template=eda.TEMPLATE,
                height=320,
                showlegend=False,
                coloraxis_showscale=False,
                title=dict(text="Probabilitas tiap kelas", font_size=14),
                xaxis_title="Probabilitas",
                yaxis_title="",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_p, width="stretch")


# ----------------------------------------------------------------------------
# 5. Data Explorer
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Telusuri data mentah")

    c1, c2 = st.columns([2, 2])
    keyword = c1.text_input("Cari (ID distribusi / provinsi)", "")
    default_cols = [
        "distribution_id",
        "province",
        "beneficiary_category",
        "protein_type",
        "transit_time_minutes",
        "food_arrival_temperature_celsius",
        "plate_waste_percentage",
        "bgn_program_compliance_tier",
        modeling.TARGET,
    ]
    chosen = c2.multiselect("Kolom yang ditampilkan", list(df.columns), default=default_cols)

    view = df
    if keyword:
        mask = (
            view["distribution_id"].str.contains(keyword, case=False, na=False)
            | view["province"].str.contains(keyword, case=False, na=False)
        )
        view = view[mask]

    st.caption(f"{loader.fmt_int(len(view))} baris")
    st.dataframe(view[chosen] if chosen else view, width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Download hasil filter (CSV)",
        data=view[chosen].to_csv(index=False).encode("utf-8") if chosen else view.to_csv(index=False).encode("utf-8"),
        file_name="mbg_filtered.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("**Lihat narasi (rationale) satu paket**")
    pick = st.selectbox(
        "Pilih distribution_id",
        view["distribution_id"].head(500).tolist(),
    )
    if pick:
        row = df_all[df_all["distribution_id"] == pick].iloc[0]
        st.write(row["cot_nutrition_logistics_rationale"])


# ----------------------------------------------------------------------------
# 6. Progress Tracker
# ----------------------------------------------------------------------------
with tab_progress:
    st.subheader("Progres pengerjaan tugas")

    state = tracker.load_progress()
    done, total, pct = tracker.summarize(state)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.progress(pct / 100)
    with c2:
        st.metric("Selesai", f"{done}/{total}")

    if state.get("updated"):
        st.caption(f"Terakhir disimpan: {state['updated']}")

    with st.form("form_progress"):
        new_state = {"items": [], "updated": state.get("updated", "")}
        for item in state["items"]:
            c1, c2, c3 = st.columns([4, 2, 4])
            c1.markdown(f"**{item['label']}**")
            status = c2.selectbox(
                "Status",
                tracker.STATUSES,
                index=tracker.STATUSES.index(item["status"])
                if item["status"] in tracker.STATUSES
                else 0,
                key=f"status_{item['id']}",
                label_visibility="collapsed",
            )
            note = c3.text_input(
                "Catatan",
                value=item["note"],
                key=f"note_{item['id']}",
                placeholder="catatan singkat...",
                label_visibility="collapsed",
            )
            new_state["items"].append(
                {"id": item["id"], "label": item["label"], "status": status, "note": note}
            )

        if st.form_submit_button("💾 Simpan progres", type="primary"):
            tracker.save_progress(new_state)
            st.success("Progres tersimpan.")
            st.rerun()

    with st.expander("Reset semua progres"):
        if st.button("Reset ke kondisi awal"):
            tracker.save_progress(tracker.default_state())
            st.success("Progres direset.")
            st.rerun()

    st.divider()
    notes = [i for i in state["items"] if i["note"]]
    if notes:
        st.markdown("**Catatan tersimpan**")
        for n in notes:
            st.markdown(f"- *{n['label']}* → {n['note']}")

st.divider()
st.caption(
    "Dashboard MBG Insight · dibuat untuk UAS Pengenalan Pola · "
    "sumber: indonesia_mbg_nationwide_enriched_2026.csv"
)
