# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# === SETUP PAGE ===
st.set_page_config(page_title="🔢 Markov Fusion Deluxe v2", layout="centered")
st.title("🔢 Markov Fusion Deluxe v2 — Fusion China & Jawa Calendar")
st.caption("Model Markov Orde-2 dengan prediksi per posisi digit dan kombinasi angka terkuat.")

# === PARAMETER ===
alpha = st.slider("Laplace α", 0.0, 2.0, 1.0, 0.1)
top_k = st.slider("Top-K Prediksi", 1, 10, 5, 1)

# === KONVERSI HARI JAWA ===
def hari_jawa(tanggal):
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    pasaran = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
    neptu_hari = [4, 3, 7, 8, 6, 9, 5]
    neptu_pasaran = [5, 9, 7, 4, 8]

    idx_hari = tanggal.weekday()  # 0=Senin
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
                        data.append(part.zfill(4))  # fokus 4 digit terakhir
        return data
    except Exception:
        return None

# === MODEL MARKOV ORDE 2 ===
def markov_order2_probabilities(data, alpha=1.0):
    """Bangun probabilitas transisi antar digit (orde 2)."""
    sequences = [list(x) for x in data]
    transitions = {}

    for seq in sequences:
        for i in range(len(seq) - 2):
            key = (seq[i], seq[i+1])
            next_digit = seq[i+2]
            if key not in transitions:
                transitions[key] = {}
            transitions[key][next_digit] = transitions[key].get(next_digit, 0) + 1

    # Normalisasi dengan Laplace smoothing
    for k in transitions:
        total = sum(transitions[k].values()) + 10 * alpha
        for d in map(str, range(10)):
            transitions[k][d] = (transitions[k].get(d, 0) + alpha) / total

    return transitions

def top5_digits(data, alpha=1.0):
    """Prediksi top-5 digit untuk posisi ribuan, ratusan, puluhan, satuan."""
    if not data or len(data) < 3:
        return {}

    transitions = markov_order2_probabilities(data, alpha)
    last = list(data[-1])
    state = (last[-2], last[-1])

    # Probabilitas digit berikut berdasarkan state terakhir
    next_probs = transitions.get(state, {str(d): 1/10 for d in range(10)})
    sorted_probs = sorted(next_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    # Untuk kesederhanaan, gunakan pendekatan independen antar posisi
    probs_by_pos = {}
    for i, pos in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
        counter = {str(d): 0 for d in range(10)}
        for seq in data:
            counter[seq[i]] += 1
        total = sum(counter.values()) + 10 * alpha
        probs = {d: (c + alpha) / total for d, c in counter.items()}
        probs_by_pos[pos] = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

    return probs_by_pos

def top5_combinations(probs_by_pos):
    """Hitung 5 kombinasi angka paling kuat dari distribusi posisi."""
    all_combos = []
    for combo in product(
        [x[0] for x in probs_by_pos["Ribuan"]],
        [x[0] for x in probs_by_pos["Ratusan"]],
        [x[0] for x in probs_by_pos["Puluhan"]],
        [x[0] for x in probs_by_pos["Satuan"]],
    ):
        p = 1
        for i, pos in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            d = combo[i]
            p *= dict(probs_by_pos[pos])[d]
        all_combos.append(("".join(combo), p))

    return sorted(all_combos, key=lambda x: x[1], reverse=True)[:5]

# === TAMPILKAN HASIL ===
def tampilkan_prediksi(file_name, label, emoji):
    data = baca_data(file_name)
    st.subheader(f"{emoji} {label}")

    if not data:
        st.text("Tidak ada data valid.")
        return

    last_num = data[-1]
    st.markdown(f"🔹 Angka terakhir: **{last_num}**")

    probs_by_pos = top5_digits(data, alpha)
    if not probs_by_pos:
        st.warning("Data tidak cukup untuk prediksi.")
        return

    st.markdown("### 📊 Top-5 Prediksi per Posisi Digit")
    cols = st.columns(4)
    for i, pos in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
        with cols[i]:
            st.markdown(f"**{pos}:**")
            for d, p in probs_by_pos[pos]:
                st.markdown(f"- {d} ({p:.2%})")

    st.markdown("### 🔮 Top-5 Kombinasi Angka Terkuat")
    combos = top5_combinations(probs_by_pos)
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

    probs_by_pos = top5_digits(gabungan, alpha)
    st.markdown("### 📊 Top-5 per Posisi Digit (Gabungan)")
    cols = st.columns(4)
    for i, pos in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
        with cols[i]:
            st.markdown(f"**{pos}:**")
            for d, p in probs_by_pos[pos]:
                st.markdown(f"- {d} ({p:.2%})")

    st.markdown("### 🔮 Top-5 Kombinasi Angka Terkuat (Gabungan)")
    combos = top5_combinations(probs_by_pos)
    for c, p in combos:
        st.markdown(f"**{c}** — {p:.2%}")
else:
    st.text("Belum ada data valid dari file A/B/C.")
