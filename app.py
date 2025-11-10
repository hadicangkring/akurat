import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import baca_data, log_prediksi, ambil_riwayat
from fusion_model import markov_predict
from calendar_tools import hari_jawa, kalender_cina

# === SETUP PAGE ===
st.set_page_config(page_title="🔢 Markov Fusion Deluxe", layout="centered")
st.title("🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar")
st.caption("Model Markov Orde-2 dengan integrasi Hari, Pasaran, dan Kalender Cina.")

# === PARAMETER MANUAL ===
alpha = st.slider("Laplace α", 0.0, 2.0, 1.0, 0.1)
order = st.slider("Orde Markov", 1, 3, 2, 1)
top_k = st.slider("Top-K Prediksi", 1, 10, 5, 1)

# === KALENDER JAWA + CINA ===
today = datetime.now()
hari_pasaran, neptu = hari_jawa(today)
shio_elemen = kalender_cina(today)

st.markdown(f"📅 **{hari_pasaran} (Neptu {neptu})**")
st.markdown(f"🌙 **Kalender Cina:** {shio_elemen}")

# === TAMPILKAN HASIL ===
def tampilkan_prediksi(file_name, label, emoji):
    data = baca_data(file_name)
    st.subheader(f"{emoji} {label}")

    if not data:
        st.text("Tidak ada data valid.")
        return

    last_num = data[-1]
    st.write(f"Angka terakhir sebelum prediksi adalah: **{last_num}**")

    pred4 = markov_predict(data, order=order, top_k=top_k, alpha=alpha)
    pred2 = [x[-2:] for x in pred4]

    st.markdown("**Prediksi 4 Digit (Top 5):**")
    st.write(", ".join(pred4))

    st.markdown("**Prediksi 2 Digit (Top 5):**")
    st.write(", ".join(pred2))

# === JALANKAN UNTUK SETIAP FILE ===
tampilkan_prediksi("data/a.csv", "File A", "📘")
tampilkan_prediksi("data/b.csv", "File B", "📗")
tampilkan_prediksi("data/c.csv", "File C", "📙")

# === PREDIKSI GABUNGAN ===
st.subheader("🧩 Gabungan Semua Data")

data_a = baca_data("data/a.csv") or []
data_b = baca_data("data/b.csv") or []
data_c = baca_data("data/c.csv") or []

gabungan = data_a + data_b + data_c

if gabungan:
    st.write(f"Total data gabungan: {len(gabungan)} entri")

    last_real = gabungan[-1]
    st.markdown(f"🎯 **Angka Real Terakhir:** `{last_real}`")

    pred4_gab = markov_predict(gabungan, order=order, top_k=top_k, alpha=alpha)
    pred2_gab = [x[-2:] for x in pred4_gab]

    if pred4_gab:
        pred_top = pred4_gab[0]
        st.markdown(f"🔮 **Prediksi Teratas:** `{pred_top}`")

        # === Simpan log hasil prediksi ===
        log_prediksi("Gabungan", pred_top, last_real)
    else:
        st.warning("Model tidak menghasilkan prediksi valid.")
        pred_top = None

    st.markdown("**📈 Prediksi 4 Digit (Top 5 Gabungan):**")
    st.write(", ".join(pred4_gab))

    st.markdown("**📉 Prediksi 2 Digit (Top 5 Gabungan):**")
    st.write(", ".join(pred2_gab))

    # === Tampilkan Riwayat ===
    st.markdown("### 🗂️ Riwayat 5 Prediksi Terakhir")
    riwayat = ambil_riwayat(5)
    if riwayat is not None:
        st.dataframe(riwayat, use_container_width=True)
    else:
        st.info("Belum ada riwayat prediksi tersimpan.")
else:
    st.text("Belum ada data valid dari file A/B/C.")
