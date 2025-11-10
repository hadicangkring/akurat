import pandas as pd
import os
from datetime import datetime

def baca_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.dropna(how="all")
        if df.empty:
            return None
        data = []
        for val in df.values.flatten():
            if isinstance(val, str) and val.strip() != "":
                val = val.strip().replace(",", "")
                for part in val.split():
                    if part.isdigit():
                        data.append(part.zfill(6))
        return data
    except Exception:
        return None


def log_prediksi(sumber, prediksi, real, file_path):
    """Simpan log prediksi ke file berbeda per sumber"""
    status = "✅ Tepat" if prediksi == real else "❌ Meleset"
    entry = {
        "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sumber": sumber,
        "prediksi_4digit": prediksi,
        "real_4digit": real,
        "status": status
    }

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(file_path, index=False)


def ambil_riwayat(file_path, n=5):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df.tail(n)
