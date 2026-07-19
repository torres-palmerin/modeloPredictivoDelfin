"""
App Web Interactiva — Prediccion de Trayectorias Academicas.
Streamlit dashboard que consume la API Lambda para predecir
el siguiente estado academico de un estudiante.

Uso:
    streamlit run app.py

Requiere:
    - API Gateway activo en produccion
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# CONFIGURACION DE LA PAGINA
# ============================================================
st.set_page_config(
    page_title='Prediccion de Trayectorias Academicas',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ============================================================
# CONFIGURACION API
# ============================================================
API_URL = 'https://hhufc4ijx5.execute-api.us-east-2.amazonaws.com/default/DelfinInferenceHandler'
API_TIMEOUT = 45

# ============================================================
# PALETA DE COLORES — DASHBOARD OSCURO PREMIUM
# ============================================================
BG_GENERAL = '#0b1326'
BG_CARD = '#1e293b'
BG_CARD_ALT = '#171f33'
BG_SIDEBAR = '#131b2e'
BORDER_SUBTLE = '#1e3a5f'

TEXTO_PRIMARY = '#f8fafc'
TEXTO_SECONDARY = '#94a3b8'
TEXTO_MUTED = '#475569'

ACCENT_BLUE = '#4d8eff'
ACCENT_GREEN = '#4edea3'
ACCENT_RED = '#f87171'
ACCENT_AMBER = '#fbbf24'
ACCENT_PURPLE = '#a78bfa'

COLORES_ESTADO = {
    'Continuo regular': ACCENT_GREEN,
    'Exclusión': ACCENT_PURPLE,
    'PAP': ACCENT_AMBER,
    'PAT': ACCENT_RED,
    'Primera vez en una carrera': ACCENT_BLUE,
}

# ============================================================
# CATEGORIAS FIJAS
# ============================================================
PROGRAMAS = [
    "ADMINISTRACION DE EMPRESAS",
    "ARQUITECTURA",
    "CIENCIA POLITICA Y RELAC INTER",
    "COMUNICACION SOCIAL",
    "CONTADURIA PUBLICA",
    "DERECHO",
    "DISEÑO",
    "ECONOMIA",
    "FINANZAS Y NEGOCIOS INTERNACIO",
    "INGENIERIA AMBIENTAL",
    "INGENIERIA BIOMEDICA",
    "INGENIERIA CIVIL",
    "INGENIERIA DE SISTEMAS",
    "INGENIERIA DE SISTEMAS Y COMPU",
    "INGENIERIA ELECTRICA",
    "INGENIERIA ELECTRONICA",
    "INGENIERIA GENERAL",
    "INGENIERIA INDUSTRIAL",
    "INGENIERIA MECANICA",
    "INGENIERIA MECATRONICA",
    "INGENIERIA NAVAL",
    "INGENIERIA QUIMICA",
    "MARKETING Y TRANSFORMACION DIG",
    "PROF CONTADURIA PUB PARA TECNO",
    "PROF IN SISTEMAS PARA TECNOLOG",
    "PSICOLOGIA",
    "TECNO OPERACION DE PLANT PETRO",
    "TECNOLOGIA EN GES CONTAB Y FIN",
    "TECNOLOGIA EN SISTEMAS",
]
ESTADOS = ["Continuo regular", "Exclusión", "PAP", "PAT", "Primera vez en una carrera"]

# ============================================================
# CACHING
# ============================================================
@st.cache_data
def cargar_categorias():
    """Retorna los mapeos de encoding a partir de las listas fijas."""
    mapa_programa = {n: c for c, n in enumerate(PROGRAMAS)}
    mapa_estado = {n: c for c, n in enumerate(ESTADOS)}
    return {
        'programas': mapa_programa,
        'estados': mapa_estado,
        'programas_orden': PROGRAMAS,
        'estados_orden': ESTADOS,
    }


@st.cache_data
def cargar_metricas():
    """Carga las metricas del modelo desde el JSON local."""
    ruta = Path('metricas_modelo.json')
    if not ruta.exists():
        return None
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# CSS GLOBAL — DARK PREMIUM
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ===== GLOBAL ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    html, body {
        background-color: #0b1326;
        color: #f8fafc;
    }
    .stApp {
        background-color: #0b1326;
    }

    /* ===== MAIN CONTENT ===== */
    section.main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #131b2e !important;
        border-right: 1px solid #1e3a5f !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarHeaderContent"] {
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetValue"],
    section[data-testid="stSidebar"] .stSelectbox p,
    section[data-testid="stSidebar"] .stSlider p {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #475569 !important;
        font-size: 0.75rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1e3a5f !important;
        opacity: 0.5;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #94a3b8;
    }

    /* ===== SELECTBOX & SLIDER ===== */
    [data-testid="stSelectbox"] > div > div {
        background-color: #171f33 !important;
        border-color: #1e3a5f !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    [data-testid="stSlider"] .stSlider label span {
        color: #94a3b8 !important;
    }
    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #f8fafc !important;
    }

    /* ===== BOTON PREDICIR ===== */
    .stButton > button {
        background: linear-gradient(135deg, #4d8eff 0%, #3b6fd9 100%) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 14px 32px !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        width: 100% !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(77, 142, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6ba1ff 0%, #4d8eff 100%) !important;
        box-shadow: 0 6px 28px rgba(77, 142, 255, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ===== HERO CARD — PREDICCION ===== */
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #171f33 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 36px 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero-card-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
    }
    .hero-card-value {
        font-size: 2.4em;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-card-sub {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
    }

    /* ===== CONFIDENCE CARD ===== */
    .confidence-card {
        background: linear-gradient(135deg, #1e293b 0%, #171f33 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 36px 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .confidence-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
    }
    .confidence-value {
        font-size: 3em;
        font-weight: 800;
        color: #4edea3;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .confidence-bar-bg {
        background: #0b1326;
        border-radius: 8px;
        height: 8px;
        margin-top: 16px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #4edea3, #4d8eff);
    }

    /* ===== PROBABILITY BAR ITEM ===== */
    .prob-item {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 14px;
    }
    .prob-label {
        min-width: 200px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #f8fafc;
        text-align: right;
    }
    .prob-bar-track {
        flex: 1;
        background: #0b1326;
        border-radius: 6px;
        height: 24px;
        overflow: hidden;
        position: relative;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    .prob-pct {
        min-width: 52px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #f8fafc;
        text-align: left;
    }

    /* ===== METRIC CARDS (TECH PANEL) ===== */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #171f33 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #4d8eff;
    }
    .metric-card-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    .metric-card-value {
        font-size: 1.9em;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .metric-card-value.green { color: #4edea3; }
    .metric-card-value.blue { color: #4d8eff; }
    .metric-card-value.amber { color: #fbbf24; }
    .metric-card-value.red { color: #f87171; }

    /* ===== SECTION HEADERS ===== */
    .dash-title {
        font-size: 1.6em;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.03em;
        margin-bottom: 2px;
    }
    .dash-subtitle {
        font-size: 0.92rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 24px;
    }
    .section-heading {
        font-size: 1.05em;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e3a5f;
    }

    /* ===== EMPTY STATE ===== */
    .empty-state {
        background: linear-gradient(135deg, #1e293b 0%, #171f33 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 60px 40px;
        text-align: center;
    }
    .empty-state-title {
        font-size: 1.1em;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .empty-state-text {
        font-size: 0.92em;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* ===== STREAMLIT METRIC OVERRIDE ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #171f33 100%) !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] span,
    [data-testid="stMetric"] [data-testid="stMetricValue"] p {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] span,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] span,
    [data-testid="stMetric"] [data-testid="stMetricDelta"] p {
        color: #4edea3 !important;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    details[open] summary {
        border-bottom: 1px solid #1e3a5f !important;
    }

    /* ===== TABS ===== */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ===== HIDE DEFAULT STREAMLIT FOOTER ===== */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# BARRA LATERAL
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; margin-bottom: 8px;">'
        '<span style="font-size:1.6em;">🎓</span>'
        '</div>'
        '<h2 style="text-align:center; color:#f8fafc; font-weight:800; '
        'font-size:1.15em; margin:0 0 2px 0;">PREDICCION ACADEMICA</h2>'
        '<p style="text-align:center; color:#475569; font-size:0.72rem; '
        'text-transform:uppercase; letter-spacing:0.1em; margin:0 0 16px 0;">'
        'Modelo predictivo · Random Forest</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr>', unsafe_allow_html=True)

    categorias = cargar_categorias()

    programas_disponibles = categorias['programas_orden']
    carrera_seleccionada = st.selectbox(
        'Carrera',
        options=programas_disponibles,
        index=0,
    )

    estados_disponibles = categorias['estados_orden']
    estado_actual_seleccionado = st.selectbox(
        'Estado Actual',
        options=estados_disponibles,
        index=0,
    )

    st.markdown('<hr>', unsafe_allow_html=True)

    ppp_valor = st.slider(
        'PPP  (Periodo)',
        min_value=0.0, max_value=5.0, value=3.5, step=0.1,
    )

    ppa_valor = st.slider(
        'PPA  (Acumulado)',
        min_value=0.0, max_value=5.0, value=3.5, step=0.1,
    )

    st.markdown('<hr>', unsafe_allow_html=True)
    predecir = st.button('⚡  Predecir Siguiente Estado')

    st.markdown(
        '<div style="margin-top:24px; text-align:center;">'
        '<span style="color:#475569; font-size:0.68rem; text-transform:uppercase; '
        'letter-spacing:0.08em;">'
        'AWS Lambda · Random Forest Classifier'
        '</span></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER PRINCIPAL
# ============================================================
st.markdown(
    '<div class="dash-title">Prediccion de Trayectorias Academicas</div>'
    '<div class="dash-subtitle">'
    'Estimacion del siguiente estado academico basado en promedio, '
    'programa y estado actual del automata.'
    '</div>',
    unsafe_allow_html=True,
)

tab_prediccion, tab_tecnico = st.tabs(['🔮  Predicción', '🛠️  Panel Técnico'])


# ============================================================
# PESTANA 1: PREDICCION
# ============================================================
with tab_prediccion:
    if predecir:
        mapa_prog = categorias['programas']
        mapa_est = categorias['estados']

        if carrera_seleccionada not in mapa_prog:
            st.error(f"Carrera no registrada: '{carrera_seleccionada}'")
            st.stop()
        if estado_actual_seleccionado not in mapa_est:
            st.error(f"Estado no registrado: '{estado_actual_seleccionado}'")
            st.stop()

        payload = {
            'PPP': float(ppp_valor),
            'PPA': float(ppa_valor),
            'ESTADO_ACTUAL_ENCODED': int(mapa_est[estado_actual_seleccionado]),
            'PROGRAMA_ENCODED': int(mapa_prog[carrera_seleccionada]),
        }

        try:
            with st.spinner('Consultando modelo en la nube...'):
                respuesta = requests.post(
                    API_URL, json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=API_TIMEOUT,
                )
                respuesta.raise_for_status()
                respuesta = respuesta.json()

            if 'error' in respuesta:
                st.error(f'Error de la API: {respuesta["error"]}')
            else:
                prediccion = respuesta['prediccion']
                certeza = respuesta['certeza']
                probabilidades = respuesta['probabilidades']

                color_accent = COLORES_ESTADO.get(prediccion, ACCENT_BLUE)

                # --- HERO + CONFIDENCE ---
                col_hero, col_conf = st.columns([3, 2])

                with col_hero:
                    st.markdown(
                        f'<div class="hero-card">'
                        f'  <div class="hero-card-label">Estado Predicho</div>'
                        f'  <div class="hero-card-value" style="color:{color_accent};">'
                        f'{prediccion.upper()}</div>'
                        f'  <div class="hero-card-sub">Siguiente estado en el automata</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with col_conf:
                    pct_bar = int(certeza * 100)
                    st.markdown(
                        f'<div class="confidence-card">'
                        f'  <div class="confidence-label">Confianza del Modelo</div>'
                        f'  <div class="confidence-value">{certeza:.1%}</div>'
                        f'  <div class="confidence-bar-bg">'
                        f'    <div class="confidence-bar-fill" style="width:{pct_bar}%;"></div>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

                # --- DISTRIBUCION DE PROBABILIDADES ---
                st.markdown(
                    '<div class="section-heading">Distribucion de Probabilidades</div>',
                    unsafe_allow_html=True,
                )

                sorted_probs = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)

                for estado, prob in sorted_probs:
                    color_bar = COLORES_ESTADO.get(estado, ACCENT_BLUE)
                    pct = int(prob * 100)
                    st.markdown(
                        f'<div class="prob-item">'
                        f'  <div class="prob-label">{estado}</div>'
                        f'  <div class="prob-bar-track">'
                        f'    <div class="prob-bar-fill" style="width:{pct}%; background:{color_bar};"></div>'
                        f'  </div>'
                        f'  <div class="prob-pct">{prob:.1%}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

                # --- TOP 3 ---
                st.markdown(
                    '<div class="section-heading">Top 3 Estados Mas Probables</div>',
                    unsafe_allow_html=True,
                )

                top3 = sorted_probs[:3]
                medals = ['🥇', '🥈', '🥉']
                medal_colors = [ACCENT_AMBER, '#c0c0c0', '#cd7f32']

                cols_top3 = st.columns(3)
                for i, (estado, prob) in enumerate(top3):
                    with cols_top3[i]:
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#1e293b,#171f33); '
                            f'border:1px solid #1e3a5f; border-radius:14px; padding:24px 16px; '
                            f'text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.25);">'
                            f'  <div style="font-size:2em; margin-bottom:6px;">{medals[i]}</div>'
                            f'  <div style="font-size:0.85rem; font-weight:600; color:#f8fafc; '
                            f'margin-bottom:10px; line-height:1.3;">{estado}</div>'
                            f'  <div style="font-size:1.7em; font-weight:800; '
                            f'color:{medal_colors[i]};">{prob:.1%}</div>'
                            f'  <div style="font-size:0.65rem; font-weight:500; color:#475569; '
                            f'text-transform:uppercase; letter-spacing:0.08em; margin-top:4px;">'
                            f'Probabilidad</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        except requests.exceptions.ConnectionError:
            st.error('No se pudo conectar con la API. Verifique su conexion a internet.')
        except requests.exceptions.Timeout:
            st.error('La API no respondio a tiempo. Intente de nuevo.')
        except requests.exceptions.HTTPError as e:
            st.error(f'Error HTTP de la API: {e.response.status_code}')
        except Exception as e:
            st.error(f'Error inesperado: {e}')

    else:
        st.markdown(
            '<div class="empty-state">'
            '  <div style="font-size:2.5em; margin-bottom:12px;">🔮</div>'
            '  <div class="empty-state-title">Configure los datos del estudiante</div>'
            '  <div class="empty-state-text">'
            '  Seleccione la carrera, estado actual y promedios en la barra lateral.<br>'
            '  Luego haga clic en <strong>Predecir</strong> para consultar el modelo.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        metricas = cargar_metricas()
        if metricas:
            st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-heading">Rendimiento del Modelo</div>',
                unsafe_allow_html=True,
            )
            val = metricas.get('validacion', metricas.get('resultado', {}))
            acc = val.get('accuracy_test', val.get('accuracy', 0))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Accuracy Test', f'{acc:.1%}')
            c2.metric('Precision (macro)', f'{val.get("precision_macro", 0):.1%}')
            c3.metric('Recall (macro)', f'{val.get("recall_macro", 0):.1%}')
            c4.metric('Muestras Test', f'{val.get("n_muestras_test", 0):,}')


# ============================================================
# PESTANA 2: PANEL TECNICO
# ============================================================
with tab_tecnico:
    metricas = cargar_metricas()

    if metricas:
        val = metricas.get('validacion', metricas.get('resultado', {}))
        res = metricas.get('resultado', val)

        # --- 4 TARJETAS DE METRICAS ---
        st.markdown(
            '<div class="section-heading">Metricas de Rendimiento</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        acc_test = val.get('accuracy_test', val.get('accuracy', 0))
        prec = res.get('precision_macro', 0)
        rec = res.get('recall_macro', 0)
        n_test = res.get('n_muestras_test', 0)

        with m1:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-card-label">Accuracy Test</div>'
                f'  <div class="metric-card-value green">{acc_test:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-card-label">Precision Macro</div>'
                f'  <div class="metric-card-value blue">{prec:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-card-label">Recall Macro</div>'
                f'  <div class="metric-card-value amber">{rec:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-card-label">Muestras Test</div>'
                f'  <div class="metric-card-value">{n_test:,}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

        # --- VALIDACION CRUZADA ---
        with st.expander('Validación Cruzada (5-Fold)', expanded=False):
            cv_mean = val.get('cross_val_mean', 0)
            cv_std = val.get('cross_val_std', 0)
            st.markdown(
                f'<div style="color:#f8fafc; font-size:0.92rem; margin-bottom:12px;">'
                f'<strong>Media:</strong> {cv_mean:.2%} &nbsp;|&nbsp; '
                f'<strong>Std:</strong> ±{cv_std:.4f}</div>',
                unsafe_allow_html=True,
            )
            cv_scores = val.get('cross_val_scores', [])
            if cv_scores:
                fold_cols = st.columns(len(cv_scores))
                for i, score in enumerate(cv_scores):
                    with fold_cols[i]:
                        st.metric(f'Fold {i+1}', f'{score:.4f}')

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # --- IMPORTANCIA DE VARIABLES ---
        with st.expander('Importancia de Variables (Feature Importance)', expanded=True):
            importancias = metricas.get('importancias_features', {})
            if importancias:
                df_imp = pd.DataFrame({
                    'Feature': importancias.keys(),
                    'Importancia': importancias.values(),
                }).sort_values('Importancia', ascending=True)

                fig_imp = go.Figure()
                fig_imp.add_trace(go.Bar(
                    y=df_imp['Feature'],
                    x=df_imp['Importancia'],
                    orientation='h',
                    marker_color=ACCENT_BLUE,
                    text=[f'{v:.4f}' for v in df_imp['Importancia']],
                    textposition='outside',
                    textfont=dict(size=12, color=TEXTO_PRIMARY, family='Inter'),
                ))
                fig_imp.update_layout(
                    xaxis=dict(
                        color=TEXTO_SECONDARY,
                        gridcolor='#1e3a5f',
                        zerolinecolor='#1e3a5f',
                        range=[0, max(importancias.values()) * 1.25],
                    ),
                    yaxis=dict(color=TEXTO_PRIMARY, tickfont=dict(size=12)),
                    height=300,
                    margin=dict(l=10, r=30, t=10, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=BG_GENERAL,
                    font=dict(family='Inter', color=TEXTO_PRIMARY),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        # --- MATRIZ DE CONFUSION ---
        with st.expander('Matriz de Confusión', expanded=False):
            cm_data = metricas.get('matriz_confusion', {})
            if cm_data:
                cm = cm_data['valores_absolutos']
                etiquetas = cm_data['etiquetas']
                df_cm = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)

                fig_cm = px.imshow(
                    df_cm, text_auto=True,
                    color_continuous_scale=[
                        [0, BG_GENERAL],
                        [0.5, '#1e3a5f'],
                        [1, ACCENT_BLUE],
                    ],
                    aspect='auto',
                    labels=dict(x='Prediccion', y='Real', color='Frecuencia'),
                )
                fig_cm.update_layout(
                    height=max(380, len(etiquetas) * 55),
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=BG_GENERAL,
                    font=dict(family='Inter', color=TEXTO_PRIMARY, size=11),
                    coloraxis_colorbar=dict(
                        tickfont=dict(color=TEXTO_SECONDARY),
                        title_font=dict(color=TEXTO_SECONDARY),
                    ),
                )
                fig_cm.update_xaxes(color=TEXTO_PRIMARY, tickfont=dict(color=TEXTO_PRIMARY, size=10))
                fig_cm.update_yaxes(color=TEXTO_PRIMARY, tickfont=dict(color=TEXTO_PRIMARY, size=10))
                st.plotly_chart(fig_cm, use_container_width=True)

        # --- REPORTE DE CLASIFICACION ---
        with st.expander('Reporte de Clasificación Completo', expanded=False):
            reporte = metricas.get('reporte_clasificacion', '')
            if reporte:
                st.code(reporte, language=None)

        # --- HIPERPARAMETROS ---
        with st.expander('Hiperparámetros del Modelo', expanded=False):
            st.json(metricas.get('hiperparametros', {}))

        # --- GRAFICOS DE EVALUACION ---
        directorio_figuras = Path('reports/figures')
        if directorio_figuras.exists():
            pngs = sorted(directorio_figuras.glob('*.png'))
            if pngs:
                st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-heading">Graficos de Evaluacion</div>',
                    unsafe_allow_html=True,
                )
                cols_graficos = st.columns(2)
                for i, png in enumerate(pngs):
                    with cols_graficos[i % 2]:
                        st.image(
                            str(png),
                            caption=png.stem.replace('_', ' ').title(),
                            use_container_width=True,
                        )
    else:
        st.warning(
            'No se encontro metricas_modelo.json. '
            'Ejecuta python run_local_training.py para generarlo.'
        )
