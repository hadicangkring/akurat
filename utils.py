import pandas as pd
import os

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

def log_activity(msg, file="activity.log"):
    with open(file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
