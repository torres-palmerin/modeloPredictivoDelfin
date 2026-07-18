"""
App Web Interactiva — Prediccion de Trayectorias Academicas.
Streamlit dashboard que consume la API Lambda para predecir
el siguiente estado academico de un estudiante.

Uso:
    streamlit run app.py

Requiere:
    - processed CSV en la raiz (para reconstruir categorias)
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
# COLORES INSTITUCIONALES
# ============================================================
COLOR_PRIMARIO = '#1B3A5C'
COLOR_SECUNDARIO = '#3A7CA5'
COLOR_ACCENTO = '#5DADE2'
COLOR_EXITO = '#27AE60'
COLOR_ADVERTENCIA = '#F39C12'
COLOR_PELIGRO = '#E74C3C'

# Paleta oscura del dashboard
FONDO_SUPERFICIE = '#1A1D24'
FONDO_TARJETA = '#1E222B'
BORDE_OSCURO = '#2D3440'
TEXTO_CLARO = '#FAFAFA'
TEXTO_SECUNDARIO = '#A0A4A8'

COLORES_ESTADO = {
    'Continuo regular': COLOR_EXITO,
    'PAP': COLOR_ADVERTENCIA,
    'PAT': COLOR_PELIGRO,
    'Exclusion': '#8E44AD',
    'Recuperacion academica': '#E67E22',
    'Primera vez en una carrera': COLOR_SECUNDARIO,
}

# ============================================================
# CACHING
# ============================================================
@st.cache_data
def cargar_categorias():
    """Reconstruye los mapeos de encoding a partir del CSV procesado."""
    csv_candidatos = list(Path('.').glob('procesado_*.csv'))
    if not csv_candidatos:
        csv_candidatos = list(Path('data').glob('*.csv'))
    if not csv_candidatos:
        return None

    df = pd.read_csv(csv_candidatos[-1])
    programa_cat = df['PROGRAMA'].astype('category')
    estado_cat = df['AUTOMATA_ESTADO_MATH'].astype('category')

    mapa_programa = {n: c for c, n in enumerate(programa_cat.cat.categories)}
    mapa_estado = {n: c for c, n in enumerate(estado_cat.cat.categories)}

    return {
        'programas': mapa_programa,
        'estados': mapa_estado,
        'programas_orden': list(programa_cat.cat.categories),
        'estados_orden': list(estado_cat.cat.categories),
        'total_registros': len(df),
        'total_estudiantes': df['ID'].nunique(),
    }


@st.cache_data
def cargar_metricas():
    """Carga las metricas del modelo desde el JSON local."""
    ruta = Path('metricas_modelo.json')
    if not ruta.exists():
        return None
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)


def llamar_api(payload: dict) -> dict:
    """Envia POST a la API Gateway y retorna la respuesta JSON."""
    respuesta = requests.post(API_URL, json=payload, timeout=API_TIMEOUT)
    respuesta.raise_for_status()
    return respuesta.json()


# ============================================================
# ESTILOS CSS — DARK MODE PROFESIONAL
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #F0F4F8;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1B3A5C !important;
        font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] span {
        color: #262730 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetValue"],
    section[data-testid="stSidebar"] .stSelectbox p,
    section[data-testid="stSidebar"] .stSlider p {
        color: #262730 !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] [data-testid="stTooltipInline"] {
        color: #555555 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #CBD5E1;
    }

    /* BOTON */
    .stButton > button {
        background-color: #1B3A5C;
        color: white;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        width: 100%;
        border: none;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #3A7CA5;
    }

    /* BANNERS */
    .banner-resultado {
        padding: 24px 32px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4em;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
        margin: 8px 0 24px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .banner-exito {
        background: linear-gradient(135deg, #1E8449 0%, #27AE60 100%);
        color: white;
    }
    .banner-advertencia {
        background: linear-gradient(135deg, #D68910 0%, #F39C12 100%);
        color: white;
    }
    .banner-peligro {
        background: linear-gradient(135deg, #C0392B 0%, #E74C3C 100%);
        color: white;
    }

    /* TARJETAS TOP-3 */
    .top3-card {
        background-color: #1E222B;
        border: 1px solid #2D3440;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        transition: border-color 0.2s, transform 0.2s;
    }
    .top3-card:hover {
        border-color: #3A7CA5;
        transform: translateY(-2px);
    }
    .top3-rank {
        font-size: 2.2em;
        font-weight: 700;
        color: #555E68;
        font-family: 'Inter', sans-serif;
        line-height: 1;
        margin-bottom: 8px;
    }
    .top3-rank-1 { color: #F39C12; }
    .top3-rank-2 { color: #A0A4A8; }
    .top3-rank-3 { color: #CD7F32; }
    .top3-nombre {
        font-size: 1.05em;
        font-weight: 600;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
        margin-bottom: 12px;
        line-height: 1.3;
    }
    .top3-porcentaje {
        font-size: 1.8em;
        font-weight: 700;
        color: #27AE60;
        font-family: 'Inter', sans-serif;
    }
    .top3-label {
        font-size: 0.75em;
        font-weight: 500;
        color: #555E68;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* SECCIONES */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.3em;
        color: #FAFAFA;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .section-subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 0.95em;
        color: #A0A4A8;
        margin-bottom: 16px;
    }

    /* METRICAS */
    [data-testid="stMetric"] {
        background-color: #1E222B;
        border: 1px solid #2D3440;
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] span,
    [data-testid="stMetric"] [data-testid="stMetricValue"] p {
        color: #FAFAFA !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] span,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #A0A4A8 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] span,
    [data-testid="stMetric"] [data-testid="stMetricDelta"] p {
        color: #FAFAFA !important;
    }

    /* PANEL TECNICO */
    .cv-card {
        background: #1E222B;
        border-left: 4px solid #1B3A5C;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .cv-card .cv-label { font-size: 0.85em; color: #A0A4A8; margin-bottom: 4px; }
    .cv-card .cv-value { font-size: 1.6em; font-weight: bold; color: #FAFAFA; }
    .cv-card .cv-sub { font-size: 0.8em; color: #555E68; }

    /* INFO BOX */
    .info-box {
        background-color: #1E222B;
        border: 1px solid #2D3440;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
    }
    .info-box p {
        color: #A0A4A8;
        font-size: 1.05em;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BARRA LATERAL
# ============================================================
with st.sidebar:
    st.markdown('# Prediccion Academica')
    st.markdown('### Ingrese los datos del estudiante')
    st.markdown('---')

    categorias = cargar_categorias()

    if categorias is None:
        st.error('No se encontro el dataset procesado para reconstruir las categorias.')
        st.stop()

    programas_disponibles = categorias['programas_orden']
    carrera_seleccionada = st.selectbox(
        'Carrera (Programa)',
        options=programas_disponibles,
        index=0,
        help='Seleccione la carrera del estudiante.',
    )

    estados_disponibles = categorias['estados_orden']
    estado_actual_seleccionado = st.selectbox(
        'Estado Academico Actual',
        options=estados_disponibles,
        index=0,
        help='Estado actual del estudiante en el automata.',
    )

    st.markdown('---')

    ppp_valor = st.slider(
        'Promedio del Periodo (PPP)',
        min_value=0.0, max_value=5.0, value=3.5, step=0.1,
        help='Promedio obtenido en el periodo mas reciente.',
    )

    ppa_valor = st.slider(
        'Promedio Acumulado (PPA)',
        min_value=0.0, max_value=5.0, value=3.5, step=0.1,
        help='Promedio acumulado historico del estudiante.',
    )

    st.markdown('---')

    predecir = st.button('Predecir Siguiente Estado')

    st.markdown('---')
    st.caption('Backend: AWS Lambda (Random Forest)')
    st.caption(f'Registros entrenados: {categorias["total_registros"]:,}')
    st.caption(f'Estudiantes: {categorias["total_estudiantes"]:,}')


# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================
st.markdown('<p class="section-title">Prediccion de Trayectorias Academicas</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">'
    'Modelo predictivo para estimar el siguiente estado academico '
    'basado en promedio, programa y estado actual del automata.'
    '</p>',
    unsafe_allow_html=True,
)

tab_prediccion, tab_tecnico = st.tabs(['Prediccion', 'Panel Tecnico'])


# ============================================================
# PESTANA 1: PREDICCION
# ============================================================
with tab_prediccion:
    if predecir:
        mapa_prog = categorias['programas']
        mapa_est = categorias['estados']

        if carrera_seleccionada not in mapa_prog:
            st.error(f"Error: La carrera '{carrera_seleccionada}' no esta registrada.")
            st.stop()
        if estado_actual_seleccionado not in mapa_est:
            st.error(f"Error: El estado '{estado_actual_seleccionado}' no esta registrado.")
            st.stop()

        valor_estado_final = int(mapa_est[estado_actual_seleccionado])
        valor_programa_final = int(mapa_prog[carrera_seleccionada])

        payload = {
            'PPP': float(ppp_valor),
            'PPA': float(ppa_valor),
            'ESTADO_ACTUAL_ENCODED': valor_estado_final,
            'PROGRAMA_ENCODED': valor_programa_final,
        }

        headers = {'Content-Type': 'application/json'}

        try:
            with st.spinner('Consultando modelo en la nube...'):
                respuesta = requests.post(API_URL, json=payload, headers=headers, timeout=API_TIMEOUT)
                respuesta.raise_for_status()
                respuesta = respuesta.json()

            if 'error' in respuesta:
                st.error(f'Error de la API: {respuesta["error"]}')
            else:
                prediccion = respuesta['prediccion']
                certeza = respuesta['certeza']
                probabilidades = respuesta['probabilidades']

                # Banner de resultado
                if 'regular' in prediccion.lower() or 'primera' in prediccion.lower():
                    clase_banner = 'exito'
                elif 'exclusion' in prediccion.lower():
                    clase_banner = 'peligro'
                elif 'pat' in prediccion.lower():
                    clase_banner = 'peligro'
                elif 'pap' in prediccion.lower():
                    clase_banner = 'advertencia'
                else:
                    clase_banner = 'advertencia'

                st.markdown(
                    f'<div class="banner-resultado banner-{clase_banner}">'
                    f'Prediccion: {prediccion} &mdash; {certeza:.1%} certeza'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Grafico de barras
                df_prob = pd.DataFrame({
                    'Estado': list(probabilidades.keys()),
                    'Probabilidad': list(probabilidades.values()),
                }).sort_values('Probabilidad', ascending=True)

                df_prob['Color'] = df_prob['Estado'].map(
                    lambda e: COLORES_ESTADO.get(e, COLOR_SECUNDARIO)
                )

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=df_prob['Estado'],
                    x=df_prob['Probabilidad'],
                    orientation='h',
                    marker_color=df_prob['Color'],
                    text=[f'{p:.1%}' for p in df_prob['Probabilidad']],
                    textposition='outside',
                    textfont=dict(size=13, color=TEXTO_CLARO, family='Inter'),
                ))
                fig.update_layout(
                    xaxis_title='Probabilidad',
                    yaxis_title='',
                    xaxis=dict(
                        tickformat='.0%', range=[0, 1],
                        color=TEXTO_SECUNDARIO,
                        gridcolor='#2D3440',
                        zerolinecolor='#2D3440',
                    ),
                    yaxis=dict(color=TEXTO_CLARO, tickfont=dict(size=12)),
                    title=dict(
                        text='Distribucion de Probabilidades por Estado',
                        font=dict(size=15, color=TEXTO_CLARO, family='Inter'),
                        x=0.0, xanchor='left',
                    ),
                    height=max(350, len(probabilidades) * 55),
                    margin=dict(l=10, r=30, t=50, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=FONDO_SUPERFICIE,
                    font=dict(family='Inter', color=TEXTO_CLARO),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Top-3
                st.markdown(
                    '<p class="section-title">Top 3 Estados Mas Probables</p>',
                    unsafe_allow_html=True,
                )

                top3 = df_prob.nlargest(3, 'Probabilidad')
                cols_top3 = st.columns(3)

                for i, (_, fila) in enumerate(top3.iterrows()):
                    rank_class = f'top3-rank top3-rank-{i+1}'
                    cols_top3[i].markdown(
                        f'<div class="top3-card">'
                        f'  <div class="{rank_class}">#{i+1}</div>'
                        f'  <div class="top3-nombre">{fila["Estado"]}</div>'
                        f'  <div class="top3-porcentaje">{fila["Probabilidad"]:.1%}</div>'
                        f'  <div class="top3-label">Probabilidad</div>'
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
            '<div class="info-box">'
            '<p>Configure los datos del estudiante en la barra lateral '
            'y haga clic en <strong>Predecir</strong> para consultar el modelo en la nube.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        metricas = cargar_metricas()
        if metricas:
            st.markdown('---')
            st.markdown(
                '<p class="section-title">Rendimiento del Modelo</p>',
                unsafe_allow_html=True,
            )
            val = metricas.get('validacion', metricas.get('resultado', {}))
            acc = val.get('accuracy_test', val.get('accuracy', 0))
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric('Accuracy Test', f'{acc:.1%}')
            col_b.metric('Precision (macro)', f'{val.get("precision_macro", 0):.1%}')
            col_c.metric('Recall (macro)', f'{val.get("recall_macro", 0):.1%}')
            col_d.metric('Muestras Test', f'{val.get("n_muestras_test", 0):,}')


# ============================================================
# PESTANA 2: PANEL TECNICO
# ============================================================
with tab_tecnico:
    st.markdown(
        '<p class="section-title">Panel Tecnico del Modelo</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-subtitle">'
        'Metricas, importancias de variables y graficos de evaluacion del modelo entrenado.'
        '</p>',
        unsafe_allow_html=True,
    )

    metricas = cargar_metricas()

    if metricas:
        val = metricas.get('validacion', metricas.get('resultado', {}))

        st.markdown(
            '<p class="section-title" style="font-size:1.1em;">Validacion Cientifica</p>',
            unsafe_allow_html=True,
        )

        cv_cols = st.columns(4)

        acc_train = val.get('accuracy_train', 0)
        with cv_cols[0]:
            st.metric('Accuracy Entrenamiento', f'{acc_train:.2%}')

        acc_test = val.get('accuracy_test', val.get('accuracy', 0))
        with cv_cols[1]:
            st.metric('Accuracy Prueba', f'{acc_test:.2%}')

        brecha = val.get('brecha_overfitting', 0)
        with cv_cols[2]:
            if brecha < 0.02:
                delta_text = 'optimo'
                delta_color = 'normal'
            elif brecha < 0.05:
                delta_text = 'aceptable'
                delta_color = 'off'
            else:
                delta_text = 'alto'
                delta_color = 'inverse'
            st.metric('Brecha Overfitting', f'{brecha:.4f}', delta=delta_text, delta_color=delta_color)

        cv_mean = val.get('cross_val_mean', 0)
        cv_std = val.get('cross_val_std', 0)
        with cv_cols[3]:
            st.metric('CV 5-Fold Media +/- Std', f'{cv_mean:.2%} +/- {cv_std:.4f}')

        cv_scores = val.get('cross_val_scores', [])
        if cv_scores:
            st.markdown('**Scores por pliegue:**')
            fold_cols = st.columns(len(cv_scores))
            for i, score in enumerate(cv_scores):
                with fold_cols[i]:
                    st.metric(f'Pliegue {i+1}', f'{score:.4f}')

        st.markdown('---')

        st.markdown(
            '<p class="section-title" style="font-size:1.1em;">Metricas de Evaluacion</p>',
            unsafe_allow_html=True,
        )
        res = metricas.get('resultado', val)
        col1, col2, col3 = st.columns(3)
        col1.metric('Precision (macro)', f'{res.get("precision_macro", 0):.1%}')
        col2.metric('Recall (macro)', f'{res.get("recall_macro", 0):.1%}')
        n_total = res.get('n_muestras_totales', res.get('n_muestras_train', 0) + res.get('n_muestras_test', 0))
        n_train = res.get('n_muestras_train', 0)
        n_test = res.get('n_muestras_test', 0)
        col3.metric('Total Muestras', f'{n_total:,}')

        col4, col5, col6 = st.columns(3)
        col4.metric('Muestras Entrenamiento', f'{n_train:,}')
        col5.metric('Muestras Prueba', f'{n_test:,}')
        col6.metric('Features', res.get('n_features', 0))

        st.markdown('---')

        with st.expander('Hiperparametros del Modelo', expanded=False):
            st.json(metricas.get('hiperparametros', {}))

        with st.expander('Importancia de Variables', expanded=True):
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
                    marker_color=COLOR_ACCENTO,
                    text=[f'{v:.4f}' for v in df_imp['Importancia']],
                    textposition='outside',
                    textfont=dict(size=11, color=TEXTO_CLARO, family='Inter'),
                ))
                fig_imp.update_layout(
                    title=dict(
                        text='Importancia de Variables (Feature Importance)',
                        font=dict(size=14, color=TEXTO_CLARO, family='Inter'),
                        x=0.0, xanchor='left',
                    ),
                    xaxis=dict(
                        color=TEXTO_SECUNDARIO,
                        gridcolor='#2D3440',
                        zerolinecolor='#2D3440',
                    ),
                    yaxis=dict(color=TEXTO_CLARO),
                    height=320,
                    margin=dict(l=10, r=30, t=50, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=FONDO_SUPERFICIE,
                    font=dict(family='Inter', color=TEXTO_CLARO),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander('Matriz de Confusion', expanded=False):
            cm_data = metricas.get('matriz_confusion', {})
            if cm_data:
                cm = cm_data['valores_absolutos']
                etiquetas = cm_data['etiquetas']
                df_cm = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)

                fig_cm = px.imshow(
                    df_cm, text_auto=True,
                    color_continuous_scale='Blues', aspect='auto',
                    labels=dict(x='Prediccion', y='Real', color='Frecuencia'),
                )
                fig_cm.update_layout(
                    height=max(400, len(etiquetas) * 60),
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=FONDO_SUPERFICIE,
                    font=dict(family='Inter', color=TEXTO_CLARO),
                )
                fig_cm.update_xaxes(color=TEXTO_CLARO, tickfont=dict(color=TEXTO_CLARO))
                fig_cm.update_yaxes(color=TEXTO_CLARO, tickfont=dict(color=TEXTO_CLARO))
                st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander('Reporte de Clasificacion Completo', expanded=False):
            st.code(metricas.get('reporte_clasificacion', ''), language=None)

        st.markdown('---')

        st.markdown(
            '<p class="section-title" style="font-size:1.1em;">Graficos de Evaluacion</p>',
            unsafe_allow_html=True,
        )
        directorio_figuras = Path('reports/figures')
        if directorio_figuras.exists():
            pngs = sorted(directorio_figuras.glob('*.png'))
            if pngs:
                cols_graficos = st.columns(2)
                for i, png in enumerate(pngs):
                    with cols_graficos[i % 2]:
                        st.image(
                            str(png),
                            caption=png.stem.replace('_', ' ').title(),
                            use_container_width=True,
                        )
            else:
                st.info('No se encontraron graficos en reports/figures/.')
        else:
            st.info('Directorio reports/figures/ no encontrado.')
    else:
        st.warning(
            'No se encontro metricas_modelo.json. '
            'Ejecuta python run_local_training.py para generarlo.'
        )
