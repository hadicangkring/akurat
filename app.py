# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# === PAGE SETUP ===
st.set_page_config(page_title="🎯 Markov Fusion 2D v2.2", layout="centered")
st.title("🎯 Markov Fusion 2D v2.2 — Fokus Puluhan & Satuan")
st.caption("Analisis Markov Orde-2 dengan fokus prediksi dua digit terakhir (2D).")

# === PARAMETER ===
alpha = st.slider("Laplace α", 0.0, 2.0, 1.0, 0.1)
top_k = st.slider("Top-K Kombinasi 2D", 1, 10, 5, 1)

# === HARI JAWA ===
def hari_jawa(tanggal):
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    pasaran = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
    neptu_hari = [4, 3, 7, 8, 6, 9, 5]
    neptu_pasaran = [5, 9, 7, 4, 8]
    idx_hari = tanggal.weekday()
    idx_pasaran = (tanggal.toordinal() + 3) % 5
    return f"{hari[idx_hari]} {pasaran[idx_pasaran]}", neptu_hari[idx_hari] + neptu_pasaran[idx_pasaran]

today = datetime.now()
hari_pasaran, neptu = hari_jawa(today)
st.markdown(f"📅 **{hari_pasaran} (Neptu {neptu})**")

# === BACA DATA ===
def baca_data(file_name):
    try:
        df = pd.read_csv(file_name, header=None)
        df = df.dropna(how="all")
        if df.empty:
            return None
        data = []
        for val in df.values.flatten():
            if isinstance(val, str) and val.strip() != "":
                val = val.strip().replace(",", "")
                for part in val.split():
                    if part.isdigit():
                        data.append(part.zfill(4))  # tetap 4 digit untuk stabilitas
        return data
    except Exception:
        return None

# === MODEL MARKOV ORDE 2 ===
def markov_order2_probabilities(data, alpha=1.0):
    transitions = {}
    for seq in data:
        for i in range(len(seq) - 2):
            key = (seq[i], seq[i+1])
            nxt = seq[i+2]
            if key not in transitions:
                transitions[key] = {}
            transitions[key][nxt] = transitions[key].get(nxt, 0) + 1
    for k in transitions:
        total = sum(transitions[k].values()) + 10 * alpha
        for d in map(str, range(10)):
            transitions[k][d] = (transitions[k].get(d, 0) + alpha) / total
    return transitions

# === TOP 3 DIGIT UNTUK 2D ===
def top_digits_2d(data, alpha=1.0, top_n=3):
    """Prediksi top-N digit untuk posisi Puluhan & Satuan saja."""
    if not data:
        return {}

    positions = ["Puluhan", "Satuan"]
    probs_by_pos = {}
    for i, pos in enumerate([-2, -1]):  # ambil 2 digit terakhir
        counter = {str(d): 0 for d in range(10)}
        for seq in data:
            counter[seq[pos]] += 1
        total = sum(counter.values()) + 10 * alpha
        probs = {d: (c + alpha) / total for d, c in counter.items()}
        probs_by_pos[positions[i]] = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return probs_by_pos

# === TOP 5 KOMBINASI 2D ===
def top_combinations_2d(probs_by_pos, top_k=5):
    all_combos = []
    for combo in product(
        [x[0] for x in probs_by_pos["Puluhan"]],
        [x[0] for x in probs_by_pos["Satuan"]],
    ):
        p = dict(probs_by_pos["Puluhan"])[combo[0]] * dict(probs_by_pos["Satuan"])[combo[1]]
        all_combos.append(("".join(combo), p))
    return sorted(all_combos, key=lambda x: x[1], reverse=True)[:top_k]

# === TAMPILKAN HASIL ===
def tampilkan_prediksi(file_name, label, emoji):
    data = baca_data(file_name)
    st.subheader(f"{emoji} {label}")

    if not data:
        st.text("Tidak ada data valid.")
        return

    last_num = data[-1]
    st.markdown(f"🔹 Angka terakhir: **{last_num}**")

    probs_by_pos = top_digits_2d(data, alpha, top_n=3)
    if not probs_by_pos:
        st.warning("Data tidak cukup untuk prediksi.")
        return

    # Tampilkan top-3 per posisi (2D)
    st.markdown("### 📊 Top-3 Prediksi per Posisi (Puluhan & Satuan)")
    cols = st.columns(2)
    for i, pos in enumerate(["Puluhan", "Satuan"]):
        with cols[i]:
            st.markdown(f"**{pos}:**")
            for d, p in probs_by_pos[pos]:
                st.markdown(f"- {d} ({p:.2%})")

    # Tampilkan top kombinasi 2D terkuat
    st.markdown("### 🔮 Top-5 Kombinasi 2D Terkuat")
    combos = top_combinations_2d(probs_by_pos, top_k)
    for c, p in combos:
        st.markdown(f"**{c}** — {p:.2%}")

# === JALANKAN UNTUK SETIAP FILE ===
tampilkan_prediksi("data/a.csv", "File A", "📘")
tampilkan_prediksi("data/b.csv", "File B", "📗")
tampilkan_prediksi("data/c.csv", "File C", "📙")

# === GABUNGAN SEMUA DATA ===
st.subheader("🧩 Gabungan Semua Data")
data_a = baca_data("data/a.csv") or []
data_b = baca_data("data/b.csv") or []
data_c = baca_data("data/c.csv") or []
gabungan = data_a + data_b + data_c

if gabungan:
    last_num = gabungan[-1]
    st.markdown(f"🔹 Angka terakhir gabungan: **{last_num}**")

    probs_by_pos = top_digits_2d(gabungan, alpha, top_n=3)
    st.markdown("### 📊 Top-3 per Posisi (Gabungan 2D)")
    cols = st.columns(2)
    for i, pos in enumerate(["Puluhan", "Satuan"]):
        with cols[i]:
            st.markdown(f"**{pos}:**")
            for d, p in probs_by_pos[pos]:
                st.markdown(f"- {d} ({p:.2%})")

    st.markdown("### 🔮 Top-5 Kombinasi 2D Terkuat (Gabungan)")
    combos = top_combinations_2d(probs_by_pos, top_k)
    for c, p in combos:
        st.markdown(f"**{c}** — {p:.2%}")
else:
    st.text("Belum ada data valid dari file A/B/C.")
