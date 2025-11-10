from datetime import datetime
import lunardate

def hari_jawa(tanggal):
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    pasaran = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
    neptu_hari = [4, 3, 7, 8, 6, 9, 5]
    neptu_pasaran = [5, 9, 7, 4, 8]

    idx_hari = tanggal.weekday()
    idx_pasaran = (tanggal.toordinal() + 3) % 5
    return f"{hari[idx_hari]} {pasaran[idx_pasaran]}", neptu_hari[idx_hari] + neptu_pasaran[idx_pasaran]

def kalender_cina(tanggal):
    lunar = lunardate.LunarDate.fromSolarDate(tanggal.year, tanggal.month, tanggal.day)
    shio = [
        "Tikus", "Kerbau", "Macan", "Kelinci", "Naga", "Ular",
        "Kuda", "Kambing", "Monyet", "Ayam", "Anjing", "Babi"
    ]
    elemen = ["Kayu", "Api", "Tanah", "Logam", "Air"]
    shio_tahun = shio[(lunar.year - 4) % 12]
    elemen_tahun = elemen[((lunar.year - 4) // 2) % 5]
    return f"{shio_tahun} — Elemen {elemen_tahun}"
