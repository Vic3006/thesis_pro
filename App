# -*- coding: utf-8 -*-
"""
================================================================================
OSTEO-AI  |  Sistema multimodal de cribado de osteoporosis (3 clases)
Dashboard de demostración para defensa de tesis (puerta cerrada) — v2 (estética)

Autores: César José Toledo Sánchez · Victor Hugo Martínez Ramírez
Tec de Monterrey CCM · Ingeniería Biomédica · 2026

DISEÑO DEFENDIBLE: ramas tabular e imagen evaluadas por SEPARADO (dos semáforos).
NO se fusiona probabilidad por paciente (cohortes no emparejadas).
DEMO_MODE=True entrena el modelo tabular al vuelo sobre patient_clean_ML.csv.
================================================================================
Ejecutar:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ CONFIG
DEMO_MODE = True   # <<< PON False EN LA DEFENSA Y CARGA TUS MODELOS REALES
CSV_PATH = "patient_clean_ML.csv"
TABULAR_MODEL_PATH = "models/tabular_model.joblib"
TABULAR_SCALER_PATH = "models/tabular_scaler.joblib"
CNN_MODEL_PATH = "models/cnn_model.keras"

CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]
COL = {"Normal": "#22c55e", "Osteopenia": "#f59e0b", "Osteoporosis": "#ef4444"}
EMO = {"Normal": "🟢", "Osteopenia": "🟡", "Osteoporosis": "🔴"}

FEATURE_ORDER = [
    'joint_pain', 'gender', 'age', 'menopause_age', 'height__meter',
    'weight_kg_', 'smoker', 'diabetic', 'hypothyroidism',
    'number_of_pregnancies', 'seizer_disorder', 'estrogen_use', 'dialysis',
    'family_history_of_osteoporosis', 'maximum_walking_distance_km', 'bmi_',
    'obesity', 'meno_registrada', 'fractura_previa', 'occupation_cod',
    'dieta_restrictiva', 'antecedente_medico'
]

st.set_page_config(page_title="OSTEO-AI · Cribado multimodal",
                   page_icon="🦴", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ ESTILO
st.markdown("""
<style>
  .stApp { background: linear-gradient(160deg,#0f172a 0%,#1e293b 100%); }
  section[data-testid="stSidebar"] { background:#0b1220; }
  h1,h2,h3,h4,h5, p, label, span, div { color:#e2e8f0; }
  .brand { display:flex; align-items:center; gap:14px; margin-bottom:4px;}
  .brand .logo { font-size:40px; }
  .brand .title { font-size:30px; font-weight:800; letter-spacing:-.5px;
                  background:linear-gradient(90deg,#38bdf8,#22c55e);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .card { background:#111c33; border:1px solid #243049; border-radius:16px;
          padding:22px; box-shadow:0 6px 24px rgba(0,0,0,.35); }
  .pill { display:inline-block; padding:4px 12px; border-radius:999px;
          font-size:12px; font-weight:700; }
  .stButton>button { background:linear-gradient(90deg,#0ea5e9,#22c55e);
          color:white; font-weight:700; border:0; border-radius:12px;
          padding:12px 0; font-size:17px; }
  .stButton>button:hover { filter:brightness(1.08); }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ MODELOS
@st.cache_resource
def get_tabular_model():
    if not DEMO_MODE:
        import joblib
        model = joblib.load(TABULAR_MODEL_PATH)
        scaler = joblib.load(TABULAR_SCALER_PATH) if os.path.exists(TABULAR_SCALER_PATH) else None
        return model, scaler
    import re
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(CSV_PATH)
    df.columns = [re.sub(r'[^a-zA-Z0-9_]+', '', c.replace(' ', '_')).strip().lower()
                  for c in df.columns]
    X = df[FEATURE_ORDER].values.astype(float)
    y = df['diagnosis_cod'].astype(int).values
    model = RandomForestClassifier(n_estimators=300, max_depth=6,
        min_samples_leaf=5, max_features='sqrt',
        class_weight='balanced', random_state=42).fit(X, y)
    return model, None

@st.cache_resource
def get_cnn_model():
    if DEMO_MODE:
        return None
    from tensorflow import keras
    return keras.models.load_model(CNN_MODEL_PATH)

def predict_tabular(f):
    model, scaler = get_tabular_model()
    x = np.array([[f[k] for k in FEATURE_ORDER]], dtype=float)
    if scaler is not None:
        x = scaler.transform(x)
    return model.predict_proba(x)[0]

def predict_image(pil_image):
    cnn = get_cnn_model()
    if cnn is None:
        arr = np.array(pil_image.convert("L").resize((64, 64)), float) / 255.0
        m = arr.mean()
        base = np.array([max(0.1, m), 0.33, max(0.1, 1 - m)])
        return base / base.sum()
    size = (299, 299) if "inception" in CNN_MODEL_PATH.lower() else (224, 224)
    img = pil_image.convert("RGB").resize(size)
    x = np.expand_dims(np.array(img) / 255.0, 0)
    return cnn.predict(x, verbose=0)[0]

# ------------------------------------------------------------------ GAUGE
def gauge_svg(proba):
    """Semáforo tipo medidor: arco coloreado + aguja a la clase ganadora."""
    idx = int(np.argmax(proba)); pred = CLASSES[idx]
    angle = -90 + idx * 90  # -90 Normal, 0 Osteopenia, +90 Osteoporosis
    rad = np.radians(angle)
    cx, cy, r = 150, 150, 110
    nx, ny = cx + r * 0.8 * np.sin(rad), cy - r * 0.8 * np.cos(rad)
    return f"""
    <svg viewBox="0 0 300 200" width="100%" style="max-width:340px;">
      <path d="M40 150 A110 110 0 0 1 95 55"  stroke="{COL['Normal']}"
            stroke-width="26" fill="none" stroke-linecap="round"/>
      <path d="M105 48 A110 110 0 0 1 195 48"  stroke="{COL['Osteopenia']}"
            stroke-width="26" fill="none" stroke-linecap="round"/>
      <path d="M205 55 A110 110 0 0 1 260 150" stroke="{COL['Osteoporosis']}"
            stroke-width="26" fill="none" stroke-linecap="round"/>
      <line x1="{cx}" y1="{cy}" x2="{nx:.0f}" y2="{ny:.0f}"
            stroke="#e2e8f0" stroke-width="5" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="9" fill="#e2e8f0"/>
      <text x="150" y="185" text-anchor="middle" fill="{COL[pred]}"
            font-size="26" font-weight="800">{pred}</text>
    </svg>"""

def semaphore(proba, title):
    idx = int(np.argmax(proba)); pred = CLASSES[idx]
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill' style='background:{COL[pred]}22;"
                f"color:{COL[pred]}'>{title}</span>", unsafe_allow_html=True)
    st.markdown(gauge_svg(proba), unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;font-size:15px;color:#94a3b8'>"
                f"Confianza {proba[idx]*100:.1f}%</div>", unsafe_allow_html=True)
    for i, c in enumerate(CLASSES):
        st.markdown(f"<span style='color:{COL[c]};font-weight:600'>{EMO[c]} {c}</span>",
                    unsafe_allow_html=True)
        st.progress(float(proba[i]), text=f"{proba[i]*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    return pred

def patient_recommendations(f, pred):
    r = []
    if pred == "Osteoporosis":
        r.append("Acude pronto con tu médico: tu perfil sugiere riesgo alto. Una densitometría (DXA) confirmará el diagnóstico.")
    elif pred == "Osteopenia":
        r.append("Riesgo intermedio. Conviene una valoración médica preventiva para evitar progresión.")
    else:
        r.append("Bajo riesgo. Mantén tus hábitos saludables.")
    if f['age'] >= 65: r.append("A partir de los 65 años el cribado periódico es especialmente recomendable.")
    if f['bmi_'] < 19: r.append("Tu IMC es bajo; un peso muy bajo se asocia a menor densidad ósea. Consulta sobre nutrición.")
    if f['smoker'] == 1: r.append("Fumar acelera la pérdida de masa ósea. Dejarlo mejora tu salud ósea.")
    if f['fractura_previa'] == 1: r.append("Una fractura previa eleva el riesgo de nuevas fracturas. Coméntalo con tu médico.")
    if f['family_history_of_osteoporosis'] == 1: r.append("Tienes antecedentes familiares; informa a tu médico para seguimiento.")
    if f['maximum_walking_distance_km'] < 1: r.append("La caminata con carga fortalece el hueso; auméntala gradualmente.")
    r.append("Una dieta rica en calcio y vitamina D apoya la salud ósea.")
    return r

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.markdown("<div class='brand'><span class='logo'>🦴</span>"
                    "<span class='title'>OSTEO-AI</span></div>", unsafe_allow_html=True)
st.sidebar.caption("Cribado multimodal de osteoporosis · Demo de tesis")
role = st.sidebar.radio("Perfil de usuario", ["👤 Paciente", "🩺 Médico"])
st.sidebar.divider()
modality = st.sidebar.radio("Datos a ingresar",
                            ["Solo datos clínicos", "Solo radiografía", "Ambos"])
if DEMO_MODE:
    st.sidebar.warning("MODO DEMO: las predicciones de imagen son simuladas. "
                       "Pon DEMO_MODE=False para usar modelos reales.")

# ------------------------------------------------------------------ FORMS
def tabular_form():
    st.markdown("### 🧬 Datos clínicos")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Edad (años)", 17, 100, 60)
        gender = 1 if st.selectbox("Sexo", ["Femenino", "Masculino"]) == "Femenino" else 0
        height = st.number_input("Estatura (m)", 1.30, 2.10, 1.60, step=0.01)
        weight = st.number_input("Peso (kg)", 30.0, 150.0, 65.0, step=0.5)
        bmi = round(weight / (height ** 2), 2); st.metric("IMC", bmi)
    with c2:
        menopause_age = st.number_input("Edad menopausia (0 si N/A)", 0, 70, 0)
        pregnancies = st.number_input("N.º embarazos", 0, 15, 0)
        estrogen = 1 if st.checkbox("Uso de estrógenos") else 0
        smoker = 1 if st.checkbox("Fumador(a)") else 0
        diabetic = 1 if st.checkbox("Diabetes") else 0
        hypo = 1 if st.checkbox("Hipotiroidismo") else 0
    with c3:
        fractura = 1 if st.checkbox("Fractura previa") else 0
        family = 1 if st.checkbox("Antec. familiar osteoporosis") else 0
        joint = 1 if st.checkbox("Dolor articular") else 0
        seizer = 1 if st.checkbox("Trastorno convulsivo") else 0
        dialysis = 1 if st.checkbox("Diálisis") else 0
        antecedente = 1 if st.checkbox("Otro antec. médico") else 0
        dieta = 1 if st.checkbox("Dieta restrictiva") else 0
    walk = st.slider("Distancia máx. de caminata (km)", 0.0, 10.0, 2.0, 0.5)
    obesity = min(3, max(0, int((bmi - 18.5) // 5) + 1)) if bmi >= 18.5 else 0
    return {'joint_pain': joint, 'gender': gender, 'age': age,
            'menopause_age': menopause_age, 'height__meter': height,
            'weight_kg_': weight, 'smoker': smoker, 'diabetic': diabetic,
            'hypothyroidism': hypo, 'number_of_pregnancies': pregnancies,
            'seizer_disorder': seizer, 'estrogen_use': estrogen,
            'dialysis': dialysis, 'family_history_of_osteoporosis': family,
            'maximum_walking_distance_km': walk, 'bmi_': bmi, 'obesity': obesity,
            'meno_registrada': 1 if menopause_age > 0 else 0,
            'fractura_previa': fractura, 'occupation_cod': 0,
            'dieta_restrictiva': dieta, 'antecedente_medico': antecedente}

def image_input():
    st.markdown("### 🩻 Radiografía de rodilla")
    up = st.file_uploader("Sube la radiografía (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up is not None:
        from PIL import Image
        img = Image.open(up); st.image(img, caption="Radiografía cargada", width=300)
        return img
    return None

# ------------------------------------------------------------------ MAIN
st.markdown("<div class='brand'><span class='logo'>🦴</span>"
            "<span class='title'>Cribado multimodal de osteoporosis</span></div>",
            unsafe_allow_html=True)
st.caption("Herramienta de apoyo a la decisión clínica · No sustituye el juicio médico ni la DXA")
st.write("")

features = tabular_form() if modality in ("Solo datos clínicos", "Ambos") else None
image = image_input() if modality in ("Solo radiografía", "Ambos") else None

st.write("")
run = st.button("▶  Evaluar riesgo", use_container_width=True)

if run:
    st.divider()
    proba_tab = predict_tabular(features) if features else None
    proba_img = predict_image(image) if image is not None else None
    if proba_tab is None and proba_img is None:
        st.error("Ingresa al menos una modalidad."); st.stop()

    if role == "👤 Paciente":
        both = proba_tab is not None and proba_img is not None
        cols = st.columns(2) if both else [st.container()]
        primary = None
        if proba_tab is not None:
            with cols[0]: primary = semaphore(proba_tab, "Datos clínicos")
        if proba_img is not None:
            with (cols[1] if both else cols[0]):
                p2 = semaphore(proba_img, "Radiografía")
                primary = primary or p2
        st.divider(); st.markdown("### 💡 Recomendaciones para ti")
        ref = features if features else {k: 0 for k in FEATURE_ORDER}
        if not features:
            ref.update({'age': 60, 'bmi_': 22, 'smoker': 0, 'fractura_previa': 0,
                        'family_history_of_osteoporosis': 0, 'maximum_walking_distance_km': 2})
        for x in patient_recommendations(ref, primary): st.markdown(f"- {x}")
        st.info("Resultado orientativo. Acude con un profesional de salud para valoración completa.")
    else:
        if proba_tab is not None and proba_img is not None:
            st.markdown("## Evaluación multimodal · ramas independientes")
            a, b = st.columns(2)
            with a: pt = semaphore(proba_tab, "Rama tabular (ML)")
            with b: pi = semaphore(proba_img, "Rama de imagen (CNN)")
            st.divider()
            if pt == pi:
                st.success(f"**Concordancia:** ambas ramas indican **{pt}**. Mayor confianza.")
            else:
                st.warning(f"**Discordancia clínica:** tabular sugiere **{pt}** e imagen **{pi}**. "
                           f"Esta divergencia es información relevante; considerar DXA confirmatoria.")
        elif proba_tab is not None:
            semaphore(proba_tab, "Rama tabular (ML)")
        else:
            semaphore(proba_img, "Rama de imagen (CNN)")
        st.divider(); st.markdown("### 📊 Detalle técnico")
        rows = []
        if proba_tab is not None: rows.append(["Tabular (ML)"] + [f"{p*100:.2f}%" for p in proba_tab])
        if proba_img is not None: rows.append(["Imagen (CNN)"] + [f"{p*100:.2f}%" for p in proba_img])
        st.table(pd.DataFrame(rows, columns=["Rama"] + CLASSES))
        if features:
            with st.expander("Ver vector de features ingresado"): st.json({k: features[k] for k in FEATURE_ORDER})
        st.caption("Recordatorio: las dos ramas se entrenaron en cohortes distintas y se reportan "
                   "por separado; no se computa una probabilidad fusionada por paciente.")
