import pandas as pd
import numpy as np
import streamlit as st
import datetime
from google import genai

# ========= BACA DATA
wfa = pd.read_csv("wfa-all.csv")
hfa = pd.read_csv("lhfa-all.csv")
wfh = pd.read_csv("wfh-all.csv")
hcfa = pd.read_csv("hcfa-all.csv")

# ========== FUNGSI Z-Score
def who_zscore(x, L, M, S):
    if L == 0:
        return np.log(x/M)/S
    return ((x / M) ** L - 1) / (L * S)

# ========== FUNGSI INDIKATOR
## BB Terhadap Usia
def calc_wfa(age, sex, weight):
    ref = wfa[
        (wfa['Usia'] == age) &
        (wfa['Gender'] == sex)
    ]
    if ref.empty:
        return None

    L, M, S = ref[["L", "M", "S"]].values[0]
    return who_zscore(weight, L, M, S)

## TB Terhadap Usia
def calc_hfa(age, sex, height):
    ref = hfa[
        (hfa['Usia'] == age) &
        (hfa['Gender'] == sex)
        ]
    if ref.empty:
        return None

    L, M, S = ref[["L", "M", "S"]].values[0]
    return who_zscore(height, L, M, S)

## BB Terhadap Panjang/Tinggi Badan
def calc_wfh(age, sex, weight, body_cm):
    # Tentukan tipe pengukuran berdasarkan usia
    m_type = "Length" if age < 24 else "Height"

    # Filter data WHO sesuai kolom dataset kamu
    ref = wfh[
        (wfh["Gender"] == sex) &
        (wfh["Pengukuran"] == m_type) &
        (wfh["Tinggi"] == round(body_cm, 1))
    ]

    if ref.empty:
        return None

    L, M, S = ref[["L", "M", "S"]].values[0]
    return who_zscore(weight, L, M, S)

## LK Berdasarkan Usia
def calc_hcfa(age, sex, hc):
    ref = hcfa[
        (hcfa['Usia'] == age) &
        (hcfa['Gender'] == sex)
    ]
    if ref.empty:
        return None
    L, M, S = ref[["L", "M", "S"]].values[0]
    return who_zscore(hc, L, M, S)

## ======= STATUS STUNTING (HFA)
def stunting_status(z):
    if z < -3:
        return "Risiko Stunting Berat"
    elif z < -2:
        return "Risiko Stunting"
    return "Normal"

## ======= EVALUASI GIZI
### Berat/Usia
def wfa_status(z):
    if z is None:
        return None
    elif z < -3 :
        return "Berat Anak Sangat Kurang"
    elif z < -2:
        return "Berat Anak Kurang"
    elif z > 3:
        return "Anak Obesitas"
    elif z > 2:
        return "Berat Badan Anak Berlebih"
    else:
        return "Berat Badan Anak Normal"

### Tinggi/Usia
def hfa_status(z):
    if z is None:
        return None
    elif z < -3:
        return "Anak Sangat Pendek"
    elif z < -2:
        return "Anak Pendek"
    elif z > 3:
        return "Anak Tinggi"
    else:
        return "Tinggi Anak Normal"

### Berat/Tinggi
def wfh_status(z):
    if z is None:
        return None
    elif z < -3:
        return "Gizi Anak Buruk"
    elif z < -2:
        return "Gizi Anak Kurang"
    elif z > 3:
        return  "Anak Obesitas"
    elif z > 2:
        return "Anak Overweight"
    else:
        return "Gizi Anak Baik/Normal"

### Lingkar Kepala/Usia
def hcfa_status(z):
    if z is None:
        return None
    elif z < -2:
        return "Anak Terindikasi Microcephaly. Berisiko keterlambatan kognitif, motorik, dan belajar jangka panjang, serta gangguan neurologis"
    elif z > 2:
        return "Anak Terindikasi Macrocephaly. Indikasi adanya hydrocephalus atau masalah genetik, memerlukan skrining dini"
    else:
        return "Lingkar Kepala Anak Normal"

## ======= SAFE ROUND
def safe_round(x):
    return round(x, 2) if x is not None else None

## ======= RISK STUNTING (%)
def stunting_risk_percent(hfa, wfa):
    score = 0

    if hfa < -2:
        score += 60
    if wfa < -2:
        score += 40
    return min(score, 100)

# ======== INTEGRASI GEMINI AI
### ======= KONFIGURASI AI
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Konfigurasi AI gagal: {e}")
    client = None

### ======= FUNGSI ANALISIS AI
def get_ai_analysis(data_anak, status_z):
    prompt = f"""
    Anda adalah Pakar Gizi Anak (Pediatrician) Berstandar WHO. Berikan analisis mendalam berdasarkan data:
    - Nama: {data_anak['name']}
    - Usia: {data_anak['age']} bulan ({data_anak['sex']})
    - Skor Weight for Age: {status_z['waz_z']} ({status_z['waz_label']})
    - Skor Height for Age: {status_z['haz_z']} ({status_z['haz_label']})
    - Skor Weight for Height: {status_z['whz_z']} ({status_z['whz_label']})
    - Skor Head Circum for Age: {status_z['hcz_z']} ({status_z['hcz_label']})
    
    Instruksi:
    Sekarang user adalah pihak posyandu desa dan orang tua.
    1. Berikan penjelasan apa arti angka tersebut bagi orang tua anak
    2. Jika ada indikasi risiko stunting atau gizi buruk, berikan saran nutrisi spesifik berdasarkan kebutuhan dan usia anak
    3. Selain nutrisi yang global, berikan saran alternatif nutrisi bagi penduduk desa tropis dengan budget terbatas
    4. Gunakan gaya bicara yang empati, ramah, hangat, namun tetap profesional
    5. Semua berdasarkan standar WHO terbaru, dilarang mengasal
    6. Berikan saran intervensi berdasarkan waktu beberapa bulan kedepan
    7. Maksimal 200 kata dan formatnya point-point agar mudah dibaca ya
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Oops. Gagal mendapatkan saran Gemini: {str(e)}"

## ========= STREAMLIT
st.image("image.jpg", caption="Anak Sehat Indonesia")
st.title("Sistem Skrining Gizi & Stunting Anak (WHO)")
st.write("Silakan masukkan hasil pengukuran yang telah dilakukan dengan tepat!")

date = st.date_input("Tanggal Pengukuran", value=None)
name = st.text_input("Nama Anak")
age = int(st.number_input("Usia (bulan)", min_value=1))
sex = st.selectbox("Gender", ["L", "P"])
weight = st.number_input("Berat Badan (kg)")
height = st.number_input("Panjang/Tinggi Badan (cm)")
hc = st.number_input("Lingkar Kepala (cm)")

if st.button("Analisis", type="primary"):
    data = {
        "date": date,
        "name": name,
        "age": age,
        "sex": sex,
        "weight": weight,
        "height": height,
        "hc": hc
    }

    date_meas = data["date"]
    waz_z = calc_wfa(data["age"], data["sex"], data["weight"])
    waz_label = wfa_status(waz_z)
    haz_z = calc_hfa(data["age"], data["sex"], data["height"])
    haz_label = hfa_status(haz_z)
    whz_z = calc_wfh(data["age"], data["sex"], data["weight"], data["height"])
    whz_label = wfh_status(whz_z)
    hcz_z = calc_hcfa(data["age"], data["sex"], data["hc"])
    hcz_label = hcfa_status(hcz_z)

    risk = stunting_risk_percent(haz_z, waz_z) if haz_z is not None and waz_z is not None else None
    status = stunting_status(haz_z) if haz_z is not None else None

    WFA = safe_round(waz_z)
    HFA = safe_round(haz_z)
    WFH = safe_round(whz_z)
    HCFA = safe_round(hcz_z)

    status_z = {
        "waz_z": WFA, "waz_label": waz_label,
        "haz_z": HFA, "haz_label": haz_label,
        "whz_z": WFH, "whz_label": whz_label,
        "hcz_z": HCFA, "hcz_label": hcz_label
    }

    st.subheader(f"Hasil Pengukuran: {data['name']}")

    st.markdown("### 📊 Hasil Antropometri")
    st.write(f"**Tanggal Pengukuran : {date_meas}**")
    st.write(f"**Z-Score Berat Badan menurut Usia (WFA)** : {WFA}")
    st.write(f"Keterangan: ")
    st.write(f"{waz_label}")
    st.write(f"**Z-Score Tinggi Badan menurut Usia (HFA)** : {HFA}")
    st.write(f"Keterangan: ")
    st.write(f"{haz_label}")
    st.write(f"**Z-Score Berat Badan menurut Tinggi/Panjang (WFH)** : {WFH}")
    st.write(f"Keterangan: ")
    st.write(f"{whz_label}")
    st.write(f"**Z-Score Lingkar Kepala menurut Usia (HCFA)** : {HCFA}")
    st.write(f"Keterangan: ")
    st.write(f"{hcz_label}")

    st.markdown("### 🩺 Interpretasi")
    st.write(f"**Risiko Stunting** : **{risk}%**")
    if status != "Normal":
        st.error(f"Status Stunting: {status}")
    else:
        st.success("Status Stunting: Tidak Stunting")

    st.markdown("---")
    st.markdown("### 🤖 Rekomendasi Gemini AI (Standard WHO)")
    with st.spinner("AI sedang menganalisis data..."):
        saran_ai = get_ai_analysis(data, status_z)
        st.info(saran_ai)

