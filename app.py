# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os
import matplotlib.pyplot as plt

# =====================
# Markov Fusion Deluxe
# v1.0 — Hybrid Markov (orde 1-3) + UI, caching, upload, logging, analytics
# =====================

BASE_DATA_DIR = "data"
os.makedirs(BASE_DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DATA_DIR, "log.csv")

# =====================
# Helpers: Kalender Jawa & Cina
# =====================

def hari_jawa(tanggal: datetime):
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    pasaran = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
    neptu_hari = [4, 3, 7, 8, 6, 9, 5]
    neptu_pasaran = [5, 9, 7, 4, 8]

    idx_hari = tanggal.weekday()  # 0=Senin
    idx_pasaran = (tanggal.toordinal() + 3) % 5
    return f"{hari[idx_hari]} {pasaran[idx_pasaran]}", neptu_hari[idx_hari] + neptu_pasaran[idx_pasaran]


def kalender_cina(tahun: int):
    shio = ["Tikus","Kerbau","Macan","Kelinci","Naga","Ular","Kuda","Kambing","Monyet","Ayam","Anjing","Babi"]
    # elemen berulang setiap 10 tahun: Kayu, Kayu, Api, Api, Tanah, Tanah, Logam, Logam, Air, Air
    elemen = ["Kayu","Api","Tanah","Logam","Air"]
    sh = shio[(tahun - 4) % 12]
    el = elemen[((tahun - 4) % 10) // 2]
    return sh, el

# =====================
# Baca / Normalisasi Data
# =====================
@st.cache_data
def baca_data(file_path: str):
    try:
        df = pd.read_csv(file_path, header=None, dtype=str)
        df = df.dropna(how="all")
        if df.empty:
            return []
        data = []
        for val in df.values.flatten():
            if isinstance(val, str) and val.strip() != "":
                val = val.strip().replace(",", "")
                for part in val.split():
                    # ambil hanya yang mengandung digit (abaikan teks)
                    s = "".join([c for c in part if c.isdigit()])
                    if s:
                        # normalisasikan ke 6 digit (zfill)
                        data.append(s.zfill(6))
        return data
    except Exception:
        return []

# Save uploaded file to data directory
def save_uploaded(file, name):
    path = os.path.join(BASE_DATA_DIR, name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path

# =====================
# Markov Transition Builders (orde 1,2,3)
# =====================

def build_transitions(data, order=2):
    transitions = {}
    seqs = [list(x) for x in data]
    for seq in seqs:
        for i in range(len(seq) - order):
            key = tuple(seq[i:i+order])
            next_digit = seq[i+order]
            transitions.setdefault(key, {})
            transitions[key][next_digit] = transitions[key].get(next_digit, 0) + 1
    return transitions


def normalize_transitions(transitions, alpha=1.0):
    norm = {}
    for k, nxt in transitions.items():
        total = sum(nxt.values()) + 10 * alpha
        norm[k] = {}
        for d in map(str, range(10)):
            norm[k][d] = (nxt.get(d, 0) + alpha) / total
    return norm

# =====================
# Predictors: order1, order2, order3 using beam search style
# =====================

def beam_predict_from_state(transitions, start_state, steps=2, beam_width=10):
    # transitions: dict mapping key-> {digit:prob}
    beams = [("".join(start_state), 1.0)]  # (sequence_str, score)
    for _ in range(steps):
        new_beams = []
        for seq_str, score in beams:
            state = tuple(seq_str[-len(start_state):])
            probs = transitions.get(state)
            if not probs:
                continue
            # take top candidates by prob
            items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            for d, p in items:
                new_beams.append((seq_str + d, score * p))
        if not new_beams:
            break
        # keep top beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams


def markov_predict_hybrid(data, top_k=5, alpha=1.0, beam_width=10, weights=(0.3,0.5,0.2), neptu_weight=1.0):
    # data: list of 6-digit strings
    if not data or len(data) < 3:
        return []

    last = list(data[-1])

    # build and normalize transitions
    t1 = normalize_transitions(build_transitions(data, order=1), alpha)
    t2 = normalize_transitions(build_transitions(data, order=2), alpha)
    t3 = normalize_transitions(build_transitions(data, order=3), alpha)

    # start states
    s1 = tuple(last[-1:])
    s2 = tuple(last[-2:])
    s3 = tuple(last[-3:]) if len(last) >= 3 else tuple(last)

    beams1 = beam_predict_from_state(t1, s1, steps=2, beam_width=beam_width)
    beams2 = beam_predict_from_state(t2, s2, steps=2, beam_width=beam_width)
    beams3 = beam_predict_from_state(t3, s3, steps=2, beam_width=beam_width)

    # combine candidates: prefer last 4 digits of each beam result
    candidates = {}

    def add_beams(beams, w):
        for seq_str, score in beams:
            # seq_str is start_state + predicted digits; take last 4 digits
            candidate = seq_str[-4:]
            candidates[candidate] = candidates.get(candidate, 0) + score * w

    add_beams(beams1, weights[0])
    add_beams(beams2, weights[1])
    add_beams(beams3, weights[2])

    # apply neptu weighting (simple multiplicative factor)
    for k in list(candidates.keys()):
        candidates[k] = candidates[k] * neptu_weight

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    results = [x[0] for x in ranked][:top_k]
    return results

# =====================
# Logging
# =====================

def simpan_log(sumber, preds):
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    header = False
    if not os.path.exists(LOG_FILE):
        header = True
    df = pd.DataFrame([{"waktu": datetime.now().isoformat(), "sumber": sumber, "prediksi": "|".join(preds)}])
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)

# =====================
# Streamlit UI
# =====================

st.set_page_config(page_title="🔢 Markov Fusion Deluxe", layout="wide")
st.title("🔢 Markov Fusion Deluxe — Fusion China & Jawa Calendar")
st.caption("Hybrid Markov (orde 1-3) + Kalender Jawa & Cina | v1.0")

# header info
today = datetime.now()
hari_pasaran, neptu = hari_jawa(today)
shio, elemen = kalender_cina(today.year)

col1, col2, col3 = st.columns([3,2,2])
with col1:
    st.markdown(f"### 📅 {hari_pasaran}  — Neptu: {neptu}")
    st.markdown(f"### 🧧 {shio} ({elemen})")
with col2:
    st.metric("Data Folder", BASE_DATA_DIR)
with col3:
    st.metric("Waktu Sekarang", today.strftime('%Y-%m-%d %H:%M:%S'))

# sidebar params & upload
with st.sidebar:
    st.header("Parameter Model")
    alpha = st.slider("Laplace α", 0.0, 2.0, 1.0, 0.1)
    beam_width = st.slider("Beam Width", 3, 50, 10, 1)
    top_k = st.slider("Top-K Prediksi", 1, 10, 5, 1)
    w1 = st.slider("Bobot Orde-1", 0.0, 1.0, 0.3, 0.05)
    w2 = st.slider("Bobot Orde-2", 0.0, 1.0, 0.5, 0.05)
    w3 = st.slider("Bobot Orde-3", 0.0, 1.0, 0.2, 0.05)
    # normalize weights
    total_w = max(w1 + w2 + w3, 1e-6)
    weights = (w1/total_w, w2/total_w, w3/total_w)

    st.write("---")
    st.header("Upload / Manage Data")
    uploaded = st.file_uploader("Upload CSV (satu kolom angka)", type=["csv"], accept_multiple_files=False)
    target_name = st.text_input("Nama file simpan (mis. d.csv)", value="d.csv")
    if uploaded is not None:
        if st.button("Simpan uploaded ke data/"):
            path = save_uploaded(uploaded, target_name)
            st.success(f"File tersimpan: {path}")

# tabs for view
tabs = st.tabs(["Prediksi Per File", "Prediksi Gabungan", "Analisis & Statistik", "Log & Download"])

# helper to render per-file panel
def render_file_panel(tab, file_name, label, emoji):
    with tab:
        st.header(f"{emoji} {label} — {file_name}")
        path = os.path.join(BASE_DATA_DIR, file_name)
        if not os.path.exists(path):
            st.info(f"File {file_name} tidak ditemukan di {BASE_DATA_DIR}.")
            return
        data = baca_data(path)
        if not data:
            st.warning("Tidak ada data valid di file ini.")
            return

        last_num = data[-1]
        st.subheader("Ringkasan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Angka Terakhir", last_num)
        c2.metric("Jumlah Baris", len(data))
        c3.metric("Top-K", top_k)

        # prediksi
        neptu_weight = 1.0 + (neptu % 10) / 50.0  # contoh: bobot tambahan ringan
        preds = markov_predict_hybrid(data, top_k=top_k, alpha=alpha, beam_width=beam_width, weights=weights, neptu_weight=neptu_weight)
        st.markdown("**Prediksi 4 Digit (Top)**")
        st.write(", ".join(preds))

        # simpan log
        if st.button(f"Simpan Log Prediksi ({file_name})"):
            simpan_log(file_name, preds)
            st.success("Prediksi disimpan ke log.")

        # detail expander
        with st.expander("🔍 Detail Prediksi & Distribusi"):
            # show table
            dfp = pd.DataFrame({"prediksi_4d": preds, "prediksi_2d": [p[-2:] for p in preds]})
            st.dataframe(dfp)

            # histogram digit akhir
            last_digits = [int(x[-1]) for x in data]
            fig, ax = plt.subplots()
            ax.hist(last_digits, bins=range(11))
            ax.set_title("Distribusi Digit Terakhir")
            ax.set_xlabel("Digit")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

            # transition heatmap (orde-2 matrix)
            t2 = build_transitions(data, order=2)
            # build matrix
            mat = np.zeros((10,10))
            for (a,b), nxt in t2.items():
                for d, cnt in nxt.items():
                    mat[int(b), int(d)] += cnt
            # normalize for display
            if mat.sum() > 0:
                fig2, ax2 = plt.subplots()
                im = ax2.imshow(mat)
                ax2.set_title("Matrix Transisi (orde-2): baris=posisi-1, kolom=posisi+1")
                st.pyplot(fig2)

        # download last N and preds
        if st.button(f"Download Prediksi ({file_name})"):
            out = io.StringIO()
            pd.DataFrame({"prediksi_4d": preds}).to_csv(out, index=False)
            out.seek(0)
            st.download_button("Download CSV", data=out.getvalue(), file_name=f"prediksi_{file_name}")

# render each file in tab 0
render_file_panel(tabs[0], "a.csv", "File A", "📘")
render_file_panel(tabs[0], "b.csv", "File B", "📗")
render_file_panel(tabs[0], "c.csv", "File C", "📙")

# === Gabungan ===
with tabs[1]:
    st.header("🧩 Prediksi Gabungan Semua File")
    files = [os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.csv')]
    combined = []
    for f in files:
        combined += baca_data(f)
    if not combined:
        st.info("Belum ada data valid di folder data/.")
    else:
        st.write(f"Total record gabungan: {len(combined)}")
        neptu_weight = 1.0 + (neptu % 10) / 50.0
        preds = markov_predict_hybrid(combined, top_k=top_k, alpha=alpha, beam_width=beam_width, weights=weights, neptu_weight=neptu_weight)
        st.markdown("**Prediksi 4 Digit (Top Gabungan)**")
        st.write(", ".join(preds))
        if st.button("Simpan Log Prediksi (Gabungan)"):
            simpan_log("gabungan", preds)
            st.success("Prediksi gabungan disimpan ke log.")

# === Analisis & Statistik ===
with tabs[2]:
    st.header("📊 Analisis & Statistik")
    # show list of files and quick charts
    files = [f for f in os.listdir(BASE_DATA_DIR) if f.endswith('.csv')]
    st.write("Files in data/: ", files)
    if st.button("Tampilkan Analisis Gabungan"):
        combined = []
        for f in files:
            combined += baca_data(os.path.join(BASE_DATA_DIR, f))
        if not combined:
            st.info("Belum ada data untuk dianalisis.")
        else:
            # frekuensi digit posisi terakhir
            pos_last = [int(x[-1]) for x in combined]
            fig, ax = plt.subplots()
            ax.hist(pos_last, bins=range(11))
            ax.set_title("Frekuensi Digit Terakhir (Gabungan)")
            st.pyplot(fig)

            # top-10 most common 4-digit
            from collections import Counter
            c4 = Counter([x[-4:] for x in combined if len(x) >= 4])
            top10 = c4.most_common(10)
            st.write(pd.DataFrame(top10, columns=["4-digit","count"]))

# === Log & Download ===
with tabs[3]:
    st.header("📁 Log & Download")
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
        st.dataframe(df_log)
        csv = df_log.to_csv(index=False)
        st.download_button("Download Log CSV", csv, file_name="markov_fusion_log.csv")
    else:
        st.info("Belum ada log. Menjalankan prediksi dan simpan log akan membuat file log.csv")


# =====================
# Footer: petunjuk singkat
# =====================
st.markdown("---")
st.markdown("**Petunjuk singkat:** Upload CSV berisi angka (satu angka per baris). Gunakan tombol `Simpan uploaded ke data/` lalu refresh tab `Prediksi Per File`. Gunakan slider di sidebar untuk tuning parameter model.")

