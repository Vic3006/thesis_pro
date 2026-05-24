# -*- coding: utf-8 -*-
"""
================================================================================
OSTEO-AI | Sistema multimodal de cribado de osteoporosis (3 clases) — v3
Dashboard para defensa de tesis (puerta cerrada).
Autores: César José Toledo Sánchez · Victor Hugo Martínez Ramírez · Tec CCM 2026

DISEÑO DEFENDIBLE: ramas tabular e imagen por SEPARADO (dos semáforos). NO se
fusiona probabilidad por paciente (cohortes no emparejadas).
DEMO_MODE=True: tabular entrena al vuelo sobre patient_clean_ML.csv (REAL);
imagen usa una SIMULACIÓN etiquetada (no es un modelo entrenado).
================================================================================
"""
import os
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ CONFIG
DEMO_MODE = True
CSV_PATH = "patient_clean_ML.csv"
TABULAR_MODEL_PATH = "models/tabular_model.joblib"
TABULAR_SCALER_PATH = "models/tabular_scaler.joblib"
CNN_MODEL_PATH = "models/cnn_model.keras"

# Credenciales del perfil Médico (cámbialas si quieres)
MEDICO_USER = "a01352187"
MEDICO_PASS = "Tesisfinalpro"

CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]
COL = {"Normal": "#22c55e", "Osteopenia": "#f59e0b", "Osteoporosis": "#ef4444"}
EMO = {"Normal": "🟢", "Osteopenia": "🟡", "Osteoporosis": "🔴"}

FEATURE_ORDER = [
    'joint_pain', 'gender', 'age', 'menopause_age', 'height__meter',
    'weight_kg_', 'smoker', 'diabetic', 'hypothyroidism',
    'number_of_pregnancies', 'seizer_disorder', 'estrogen_use', 'dialysis',
    'family_history_of_osteoporosis', 'maximum_walking_distance_km', 'bmi_',
    'obesity', 'meno_registrada', 'fractura_previa', 'occupation_cod',
    'dieta_restrictiva', 'antecedente_medico']

# Etiquetas legibles para el panel técnico
FEATURE_LABELS = {
    'joint_pain': 'Dolor articular', 'gender': 'Sexo (F=1)', 'age': 'Edad',
    'menopause_age': 'Edad menopausia', 'height__meter': 'Estatura (m)',
    'weight_kg_': 'Peso (kg)', 'smoker': 'Fumador', 'diabetic': 'Diabetes',
    'hypothyroidism': 'Hipotiroidismo', 'number_of_pregnancies': 'Embarazos',
    'seizer_disorder': 'Trastorno convulsivo', 'estrogen_use': 'Uso estrógenos',
    'dialysis': 'Diálisis', 'family_history_of_osteoporosis': 'Antec. familiar',
    'maximum_walking_distance_km': 'Caminata máx (km)', 'bmi_': 'IMC',
    'obesity': 'Obesidad (0-3)', 'meno_registrada': 'Menopausia registrada',
    'fractura_previa': 'Fractura previa', 'occupation_cod': 'Ocupación (cod)',
    'dieta_restrictiva': 'Dieta restrictiva', 'antecedente_medico': 'Antec. médico'}

# Métricas reales por modelo (de tu Cap. 5, out-of-fold) para el panel técnico
TABULAR_METRICS = {
    "Modelo (demo: Random Forest)": {
        "Bal. Accuracy": "71.3%", "Macro-F1": "67.0%",
        "Macro-AUC": "82.8%", "Sens. Osteoporosis": "65.3%"}}

st.set_page_config(page_title="OSTEO-AI · Cribado multimodal",
                   page_icon="🦴", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ ESTILO
st.markdown("""
<style>
  .stApp { background: linear-gradient(160deg,#0f172a 0%,#1e293b 100%); }
  section[data-testid="stSidebar"] { background:#0b1220; }
  h1,h2,h3,h4,h5,p,label,span,div { color:#e2e8f0; }
  .brand { display:flex; align-items:center; gap:14px; margin-bottom:4px; }
  .brand .logo { font-size:40px; }
  .brand .title { font-size:30px; font-weight:800; letter-spacing:-.5px;
      background:linear-gradient(90deg,#38bdf8,#22c55e);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .card { background:#111c33; border:1px solid #243049; border-radius:16px;
      padding:22px; box-shadow:0 6px 24px rgba(0,0,0,.35); }
  .pill { display:inline-block; padding:4px 12px; border-radius:999px;
      font-size:12px; font-weight:700; }
  .metric-box { background:#0e1830; border:1px solid #243049; border-radius:12px;
      padding:14px; text-align:center; }
  .metric-box .v { font-size:24px; font-weight:800; }
  .metric-box .l { font-size:12px; color:#94a3b8; }
  .role-card { background:#111c33; border:1px solid #243049; border-radius:20px;
      padding:34px 26px; text-align:center; }
  .role-card .ico { font-size:58px; }
  .role-card .rt { font-size:22px; font-weight:800; margin-top:8px; }
  .role-card .rd { font-size:14px; color:#94a3b8; margin-top:6px; }
  .stButton>button { background:linear-gradient(90deg,#0ea5e9,#22c55e);
      color:white; font-weight:700; border:0; border-radius:12px;
      padding:12px 0; font-size:17px; }
  .stButton>button:hover { filter:brightness(1.08); }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ ESTADO
if "role" not in st.session_state:
    st.session_state.role = None
if "medico_auth" not in st.session_state:
    st.session_state.medico_auth = False

def go_home():
    st.session_state.role = None
    st.session_state.medico_auth = False

# ------------------------------------------------------------------ LANDING
def landing():
    st.markdown("<div class='brand' style='justify-content:center;margin-top:30px'>"
                "<span class='logo' style='font-size:54px'>🦴</span>"
                "<span class='title' style='font-size:40px'>OSTEO-AI</span></div>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:17px'>"
                "Sistema de cribado multimodal de osteoporosis</p>",
                unsafe_allow_html=True)
    st.write(""); st.write("")
    st.markdown("<h3 style='text-align:center'>¿Cómo deseas ingresar?</h3>",
                unsafe_allow_html=True)
    st.write("")
    c1, _, c2 = st.columns([1, 0.15, 1])
    with c1:
        st.markdown("<div class='role-card'><div class='ico'>👤</div>"
                    "<div class='rt'>Paciente</div>"
                    "<div class='rd'>Vista simplificada con resultado tipo semáforo "
                    "y recomendaciones personalizadas.</div></div>", unsafe_allow_html=True)
        st.write("")
        if st.button("Entrar como Paciente", use_container_width=True, key="b_pac"):
            st.session_state.role = "👤 Paciente"; st.rerun()
    with c2:
        st.markdown("<div class='role-card'><div class='ico'>🩺</div>"
                    "<div class='rt'>Médico</div>"
                    "<div class='rd'>Vista técnica: probabilidades por clase, "
                    "concordancia entre ramas y detalle del modelo. Requiere acceso.</div></div>",
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Entrar como Médico", use_container_width=True, key="b_med"):
            st.session_state.role = "🩺 Médico"; st.rerun()
    st.write(""); st.write("")
    st.caption("Herramienta de apoyo a la decisión clínica · No sustituye el juicio "
               "médico ni la DXA · Demo de tesis")

# ------------------------------------------------------------------ LOGIN MÉDICO
def login_medico():
    st.markdown("<div class='brand' style='justify-content:center;margin-top:40px'>"
                "<span class='logo' style='font-size:46px'>🩺</span>"
                "<span class='title' style='font-size:32px'>Acceso Médico</span></div>",
                unsafe_allow_html=True)
    st.write("")
    _, c, _ = st.columns([1, 1.2, 1])
    with c:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        user = st.text_input("Usuario / ID", key="login_user")
        pwd = st.text_input("Contraseña", type="password", key="login_pwd")
        if st.button("Ingresar", use_container_width=True, key="do_login"):
            if user.strip() == MEDICO_USER and pwd == MEDICO_PASS:
                st.session_state.medico_auth = True; st.rerun()
            else:
                st.error("Credenciales incorrectas.")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("← Volver al inicio", use_container_width=True, key="back_login"):
            go_home(); st.rerun()

# Enrutamiento de acceso
if st.session_state.role is None:
    landing(); st.stop()
if st.session_state.role == "🩺 Médico" and not st.session_state.medico_auth:
    login_medico(); st.stop()
role = st.session_state.role

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
    model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced', random_state=42).fit(X, y)
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
    """Rama de imagen. En DEMO no hay CNN: se genera una distribución
    PSEUDOALEATORIA pero DETERMINISTA por imagen (mismo archivo -> mismo
    resultado, archivos distintos -> resultados distintos). NO es un modelo
    entrenado; es solo para ilustrar el flujo de la interfaz."""
    cnn = get_cnn_model()
    if cnn is None:
        # hash del contenido de la imagen -> semilla -> distribución estable
        buf = np.array(pil_image.convert("L").resize((64, 64)))
        h = int(hashlib.md5(buf.tobytes()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(h)
        p = rng.dirichlet([2.0, 1.5, 2.0])  # distribución plausible variada
        return p
    size = (299, 299) if "inception" in CNN_MODEL_PATH.lower() else (224, 224)
    img = pil_image.convert("RGB").resize(size)
    x = np.expand_dims(np.array(img) / 255.0, 0)
    return cnn.predict(x, verbose=0)[0]

# ------------------------------------------------------------------ GAUGE
def gauge_svg(proba):
    idx = int(np.argmax(proba)); pred = CLASSES[idx]
    angle = -90 + idx * 90
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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
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

def clinical_risk_flags(f):
    """Factores de riesgo detectados, para la vista médico."""
    flags = []
    if f['age'] >= 65: flags.append(("Edad ≥ 65 años", "alto"))
    elif f['age'] >= 50: flags.append(("Edad 50-64 años", "moderado"))
    if f['gender'] == 1 and f['meno_registrada'] == 1: flags.append(("Mujer posmenopáusica", "alto"))
    if f['bmi_'] < 19: flags.append(("IMC bajo (< 19)", "alto"))
    if f['fractura_previa'] == 1: flags.append(("Fractura previa", "alto"))
    if f['family_history_of_osteoporosis'] == 1: flags.append(("Antecedente familiar", "moderado"))
    if f['smoker'] == 1: flags.append(("Tabaquismo", "moderado"))
    if f['hypothyroidism'] == 1: flags.append(("Hipotiroidismo", "moderado"))
    if f['estrogen_use'] == 0 and f['gender'] == 1 and f['meno_registrada'] == 1:
        flags.append(("Sin terapia estrogénica", "moderado"))
    if f['maximum_walking_distance_km'] < 1: flags.append(("Sedentarismo", "moderado"))
    if not flags: flags.append(("Sin factores de riesgo mayores detectados", "bajo"))
    return flags

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.markdown("<div class='brand'><span class='logo'>🦴</span>"
                    "<span class='title'>OSTEO-AI</span></div>", unsafe_allow_html=True)
st.sidebar.caption("Cribado multimodal de osteoporosis · Demo de tesis")
st.sidebar.markdown(f"**Perfil activo:** {role}")
if st.sidebar.button("← Cambiar de perfil", use_container_width=True):
    go_home(); st.rerun()
st.sidebar.divider()
modality = st.sidebar.radio("Datos a ingresar",
                            ["Solo datos clínicos", "Solo radiografía", "Ambos"])
if DEMO_MODE:
    st.sidebar.warning("MODO DEMO: la rama tabular es real (entrenada sobre el "
                       "dataset clínico). La rama de imagen es una SIMULACIÓN "
                       "(no es una CNN entrenada). Conecta tu modelo y pon "
                       "DEMO_MODE=False para usarla de verdad.")

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

    # ----------------- VISTA PACIENTE -----------------
    if role == "👤 Paciente":
        both = proba_tab is not None and proba_img is not None
        cols = st.columns(2) if both else [st.container()]
        primary = None
        if proba_tab is not None:
            with cols[0]: primary = semaphore(proba_tab, "Datos clínicos")
        if proba_img is not None:
            with (cols[1] if both else cols[0]):
                p2 = semaphore(proba_img, "Radiografía"); primary = primary or p2
        st.divider(); st.markdown("### 💡 Recomendaciones para ti")
        ref = features if features else {k: 0 for k in FEATURE_ORDER}
        if not features:
            ref.update({'age': 60, 'bmi_': 22, 'smoker': 0, 'fractura_previa': 0,
                        'family_history_of_osteoporosis': 0, 'maximum_walking_distance_km': 2})
        for x in patient_recommendations(ref, primary): st.markdown(f"- {x}")
        st.info("Resultado orientativo. Acude con un profesional de salud para valoración completa.")

    # ----------------- VISTA MÉDICO -----------------
    else:
        if proba_tab is not None and proba_img is not None:
            st.markdown("## Evaluación multimodal · ramas independientes")
            a, b = st.columns(2)
            with a: pt = semaphore(proba_tab, "Rama tabular (ML)")
            with b: pi = semaphore(proba_img, "Rama de imagen (CNN)")
            st.divider()
            if pt == pi:
                st.success(f"**Concordancia entre modalidades:** ambas indican **{pt}**. Mayor confianza.")
            else:
                st.warning(f"**Discordancia clínica:** tabular sugiere **{pt}** e imagen **{pi}**. "
                           f"Divergencia relevante; considerar DXA confirmatoria.")
        elif proba_tab is not None:
            pt = semaphore(proba_tab, "Rama tabular (ML)")
        else:
            pi = semaphore(proba_img, "Rama de imagen (CNN)")

        # ---- Panel técnico enriquecido ----
        st.divider(); st.markdown("### 📊 Panel técnico")

        # Probabilidades
        rows = []
        if proba_tab is not None: rows.append(["Tabular (ML)"] + [f"{p*100:.2f}%" for p in proba_tab])
        if proba_img is not None: rows.append(["Imagen (CNN)"] + [f"{p*100:.2f}%" for p in proba_img])
        st.markdown("**Probabilidades por clase**")
        st.table(pd.DataFrame(rows, columns=["Rama"] + CLASSES))

        # Métricas de desempeño del modelo tabular (de tu Cap. 5)
        if proba_tab is not None:
            st.markdown("**Desempeño del modelo tabular** (validación 5-fold, Cap. 5)")
            m = list(TABULAR_METRICS.values())[0]
            mc = st.columns(4)
            for col, (k, v) in zip(mc, m.items()):
                col.markdown(f"<div class='metric-box'><div class='v'>{v}</div>"
                             f"<div class='l'>{k}</div></div>", unsafe_allow_html=True)

        # Factores de riesgo clínico detectados
        if features is not None:
            st.write("")
            st.markdown("**Factores de riesgo clínico detectados**")
            fc = clinical_risk_flags(features)
            color_map = {"alto": "#ef4444", "moderado": "#f59e0b", "bajo": "#22c55e"}
            chips = " ".join(
                f"<span class='pill' style='background:{color_map[lvl]}22;"
                f"color:{color_map[lvl]};margin:3px'>{txt}</span>" for txt, lvl in fc)
            st.markdown(chips, unsafe_allow_html=True)

        # Confianza y margen de decisión
        if proba_tab is not None:
            st.write("")
            srt = np.sort(proba_tab)[::-1]
            margin = (srt[0] - srt[1]) * 100
            entropy = -np.sum(proba_tab * np.log(proba_tab + 1e-9))
            cc = st.columns(3)
            cc[0].markdown(f"<div class='metric-box'><div class='v'>{srt[0]*100:.1f}%</div>"
                           f"<div class='l'>Confianza (clase top)</div></div>", unsafe_allow_html=True)
            cc[1].markdown(f"<div class='metric-box'><div class='v'>{margin:.1f} pts</div>"
                           f"<div class='l'>Margen 1ª vs 2ª clase</div></div>", unsafe_allow_html=True)
            cc[2].markdown(f"<div class='metric-box'><div class='v'>{entropy:.2f}</div>"
                           f"<div class='l'>Entropía (incertidumbre)</div></div>", unsafe_allow_html=True)
            if margin < 10:
                st.warning("Margen de decisión bajo (< 10 pts): predicción de baja "
                           "certeza. Se recomienda valoración clínica adicional.")

        # Vector de features
        if features:
            with st.expander("Ver vector de features ingresado (22 variables)"):
                tab = pd.DataFrame(
                    [(FEATURE_LABELS[k], features[k]) for k in FEATURE_ORDER],
                    columns=["Variable", "Valor"])
                st.dataframe(tab, use_container_width=True, hide_index=True)

        st.caption("Recordatorio metodológico: las dos ramas se entrenaron en cohortes "
                   "distintas y se reportan por separado; no se computa una probabilidad "
                   "fusionada por paciente. En modo demo la rama de imagen es simulada.")
