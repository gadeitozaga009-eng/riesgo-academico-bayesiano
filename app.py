
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modelo import (
    construir_sistema,
    preparar_entrada,
    inferir,
    CLASES
)


# ============================================
# CONFIGURACION DE PAGINA
# ============================================

st.set_page_config(
    page_title="SIRA | Riesgo Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# IDENTIDAD VISUAL
# ============================================

AZUL = "#173B75"
MARINO = "#101D35"
TURQUESA = "#0F9385"
DORADO = "#E5A53B"
ROJO = "#D65B64"

COLORES = {
    "Bajo": TURQUESA,
    "Medio": DORADO,
    "Alto": ROJO
}


st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background: #F4F7FC;
}

.block-container {
    max-width: 1190px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

/* PANEL LATERAL */

section[data-testid="stSidebar"] {
    background: #101D35;
}

section[data-testid="stSidebar"] * {
    color: #E7EEFA;
}

section[data-testid="stSidebar"] hr {
    border-color: #334667;
}

/* ENCABEZADO */

.hero {
    background: linear-gradient(
        120deg,
        #101D35 0%,
        #173B75 68%,
        #2461A7 100%
    );
    border-radius: 22px;
    padding: 38px 40px;
    margin-bottom: 24px;
    box-shadow: 0px 12px 30px rgba(16,29,53,0.14);
}

.hero-label {
    color: #B8D4FB;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    line-height: 1.22;
    margin-top: 12px;
}

.hero-subtitle {
    color: #D5E3F6;
    font-size: 15px;
    margin-top: 12px;
    max-width: 760px;
    line-height: 1.7;
}

.hero-tag {
    display: inline-block;
    padding: 7px 13px;
    margin-top: 17px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.17);
    border-radius: 40px;
    color: white;
    font-size: 12px;
    font-weight: 600;
}

/* TITULOS */

.section-label {
    color: #587293;
    font-weight: 800;
    font-size: 11px;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin-top: 13px;
}

.section-title {
    color: #101D35;
    font-size: 25px;
    font-weight: 800;
    margin: 8px 0 14px 0;
}

/* FORMULARIOS */

div[data-testid="stForm"] {
    background: white;
    padding: 25px;
    border: 1px solid #DFE7F2;
    border-radius: 18px;
    box-shadow: 0 5px 18px rgba(18,37,71,0.04);
}

div[data-testid="stNumberInput"] input {
    background: #F8FAFD;
}

div[data-testid="stSelectbox"] > div {
    border-radius: 10px;
}

div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {
    background: #173B75;
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 750;
    min-height: 47px;
}

div.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #24539A;
    color: white;
}

/* METRICAS */

div[data-testid="stMetric"] {
    padding: 20px;
    background: #FFFFFF;
    border: 1px solid #DFE7F2;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(18,37,71,0.035);
}

div[data-testid="stMetricLabel"] {
    color: #607593;
    font-weight: 650;
}

div[data-testid="stMetricValue"] {
    color: #101D35;
    font-weight: 800;
}

/* TARJETAS PROPIAS */

.result-card {
    background: white;
    border: 1px solid #DFE7F2;
    border-radius: 18px;
    padding: 24px;
    margin-top: 15px;
}

.result-heading {
    font-size: 12px;
    letter-spacing: 1.4px;
    color: #617795;
    font-weight: 800;
    text-transform: uppercase;
}

.result-value {
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.small-note {
    color: #687D99;
    font-size: 13px;
    line-height: 1.7;
}

.footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #DCE5F0;
    color: #617795;
    font-size: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# CARGAR Y ENTRENAR MODELO
# ============================================

@st.cache_resource(show_spinner=False)
def cargar_sistema():
    return construir_sistema()


try:
    with st.spinner(
        "Preparando la red bayesiana y las métricas..."
    ):
        sistema = cargar_sistema()

except Exception as error:
    st.error(
        "No fue posible preparar el modelo. "
        "Revisa los archivos y sus columnas."
    )

    st.exception(error)
    st.stop()


motor = sistema["motor"]

n_total = len(sistema["datos"])
n_train = sistema["n_train"]
n_val = sistema["n_val"]


# ============================================
# PANEL LATERAL
# ============================================

with st.sidebar:

    st.markdown(
        "## 🎓 SIRA"
    )

    st.caption(
        "SISTEMA INTELIGENTE DE RIESGO ACADÉMICO"
    )

    st.divider()

    st.markdown("**Institución**")
    st.write("Universidad Nacional de Trujillo")

    st.markdown("**Asignatura**")
    st.write("Inteligencia Artificial II")

    st.markdown("**Año de investigación**")
    st.write("2026")

    st.divider()

    st.markdown("### Estado del sistema")

    st.success("Modelo disponible")

    st.write(f"Registros: **{n_total}**")
    st.write("Predictores: **7**")
    st.write("Categorías: **3**")
    st.write("Arquitectura: **Naive Bayes**")

    st.divider()

    st.caption(
        "Prototipo académico de inferencia probabilística."
    )


# ============================================
# HERO
# ============================================

st.markdown("""
<div class="hero">
    <div class="hero-label">
        INTELIGENCIA ARTIFICIAL APLICADA A EDUCACIÓN
    </div>
    <div class="hero-title">
        Evaluación inteligente<br>
        del riesgo académico
    </div>
    <div class="hero-subtitle">
        Sistema basado en redes bayesianas para
        analizar factores académicos y generar
        estimaciones probabilísticas interpretables.
    </div>
    <div class="hero-tag">
        SIRA 2026 &nbsp;|&nbsp; RED BAYESIANA
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================
# INDICADORES GENERALES
# ============================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Registros analizados", n_total)

with c2:
    st.metric("Entrenamiento", n_train)

with c3:
    st.metric("Validación", n_val)

with c4:
    st.metric(
        "Exactitud de validación",
        f"{sistema['metricas_val']['Exactitud']:.1%}"
    )


# ============================================
# NAVEGACION PRINCIPAL
# ============================================

tab1, tab2, tab3 = st.tabs([
    "🎓 Evaluación académica",
    "📊 Panel técnico",
    "🔬 Metodología"
])


# ============================================
# TAB 1 - EVALUACION
# ============================================

with tab1:

    st.markdown(
        '<div class="section-label">Módulo 01</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Evaluación personalizada del estudiante'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Ingresa la información académica solicitada. "
        "El sistema procesará los datos mediante el "
        "modelo probabilístico entrenado."
    )

    with st.form("formulario_evaluacion"):

        st.markdown("#### Rendimiento académico")

        col1, col2 = st.columns(2)

        with col1:
            promedio = st.number_input(
                "Promedio académico",
                min_value=0.0,
                max_value=20.0,
                value=13.0,
                step=0.1
            )

        with col2:
            asistencia = st.number_input(
                "Asistencia a clases (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=0.5
            )

        col3, col4 = st.columns(2)

        with col3:
            desaprobados = st.number_input(
                "Cursos desaprobados",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

        with col4:
            horas = st.number_input(
                "Horas de estudio semanales",
                min_value=0,
                max_value=100,
                value=12,
                step=1
            )

        st.markdown("#### Comportamiento académico")

        col5, col6 = st.columns(2)

        with col5:
            habitos = st.selectbox(
                "Hábitos de estudio",
                [
                    "Adecuado",
                    "Regular",
                    "Inadecuado"
                ]
            )

        with col6:
            estres = st.selectbox(
                "Nivel de estrés académico",
                [
                    "Bajo",
                    "Moderado",
                    "Alto"
                ]
            )

        participacion = st.selectbox(
            "Participación académica",
            [
                "Baja",
                "Media",
                "Alta"
            ]
        )

        enviado = st.form_submit_button(
            "Evaluar riesgo académico",
            use_container_width=True,
            type="primary"
        )

    if enviado:

        evidencia = preparar_entrada(
            promedio,
            asistencia,
            desaprobados,
            horas,
            habitos,
            estres,
            participacion
        )

        nivel, probabilidades = inferir(
            motor,
            evidencia
        )

        st.session_state["resultado"] = {
            "nivel": nivel,
            "probabilidades": probabilidades
        }

    if "resultado" in st.session_state:

        resultado_actual = st.session_state[
            "resultado"
        ]

        nivel = resultado_actual["nivel"]

        probabilidades = resultado_actual[
            "probabilidades"
        ]

        color = COLORES[nivel]

        st.divider()

        st.markdown(
            '<div class="section-label">'
            'Resultado de inferencia'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">'
            'Resultado de la evaluación'
            '</div>',
            unsafe_allow_html=True
        )

        porcentaje = (
            probabilidades[nivel] * 100
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-heading">
                    Clasificación probabilística
                </div>
                <div class="result-value"
                     style="color:{color}">
                    Riesgo {nivel.lower()}
                </div>
                <div class="small-note">
                    Probabilidad posterior:
                    <strong>{porcentaje:.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        grafico = go.Figure(
            data=[
                go.Pie(
                    labels=CLASES,
                    values=[
                        probabilidades[c]
                        for c in CLASES
                    ],
                    hole=0.68,
                    marker=dict(
                        colors=[
                            COLORES[c]
                            for c in CLASES
                        ]
                    ),
                    textinfo="percent",
                    sort=False
                )
            ]
        )

        grafico.update_layout(
            title="Distribución posterior del riesgo",
            height=360,
            margin=dict(
                l=20, r=20, t=65, b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Arial",
                color="#101D35"
            ),
            legend=dict(
                orientation="h",
                y=-0.1
            )
        )

        col_res1, col_res2 = st.columns(
            [1.25, 1]
        )

        with col_res1:
            st.plotly_chart(
                grafico,
                use_container_width=True
            )

        with col_res2:

            st.markdown("#### Probabilidad por clase")

            for clase in CLASES:

                valor = probabilidades[clase]

                st.markdown(
                    f"**Riesgo {clase.lower()}** "
                    f"— {valor:.2%}"
                )

                st.progress(valor)

            st.caption(
                "Las probabilidades corresponden "
                "a la inferencia del modelo."
            )

        if nivel == "Bajo":
            st.success(
                "La clasificación estimada es baja. "
                "Se recomienda mantener un seguimiento "
                "regular del desempeño académico."
            )

        elif nivel == "Medio":
            st.warning(
                "La clasificación estimada es media. "
                "Conviene revisar los factores académicos "
                "y solicitar orientación si corresponde."
            )

        else:
            st.error(
                "La clasificación estimada es alta. "
                "Se recomienda considerar acompañamiento "
                "académico y revisar los indicadores "
                "de desempeño."
            )

        st.info(
            "Este resultado es una estimación "
            "probabilística. No constituye un "
            "diagnóstico definitivo ni determina "
            "por sí solo el futuro del estudiante."
        )

    else:
        st.caption(
            "El resultado aparecerá aquí después "
            "de completar el formulario."
        )


# ============================================
# TAB 2 - PANEL TECNICO
# ============================================

with tab2:

    st.markdown(
        '<div class="section-label">Módulo 02</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Analítica y evaluación del modelo'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Resultados calculados a partir del conjunto "
        "de entrenamiento y del subconjunto de "
        "validación, utilizando una partición "
        "estratificada del 80/20."
    )

    st.markdown("#### Indicadores de validación")

    metricas = sistema["metricas_val"]

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Exactitud",
            f"{metricas['Exactitud']:.2%}"
        )

    with b:
        st.metric(
            "Precisión macro",
            f"{metricas['Precision macro']:.2%}"
        )

    with c:
        st.metric(
            "Recall macro",
            f"{metricas['Recall macro']:.2%}"
        )

    with d:
        st.metric(
            "F1 macro",
            f"{metricas['F1 macro']:.2%}"
        )

    st.markdown("#### Comparación de desempeño")

    tabla = pd.DataFrame({
        "Indicador": list(metricas.keys()),
        "Entrenamiento": [
            sistema["metricas_train"][k]
            for k in metricas
        ],
        "Validación": [
            sistema["metricas_val"][k]
            for k in metricas
        ]
    })

    st.dataframe(
        tabla.style.format({
            "Entrenamiento": "{:.2%}",
            "Validación": "{:.2%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("#### Matriz de confusión")

    matriz = sistema["matriz"]

    figura_matriz = go.Figure(
        data=go.Heatmap(
            z=matriz,
            x=CLASES,
            y=CLASES,
            colorscale=[
                [0.0, "#E7EFF9"],
                [1.0, "#173B75"]
            ],
            text=matriz,
            texttemplate="%{text}",
            hovertemplate=(
                "Real: %{y}<br>"
                "Predicho: %{x}<br>"
                "Registros: %{z}<extra></extra>"
            ),
            showscale=True
        )
    )

    figura_matriz.update_layout(
        xaxis_title="Clasificación predicha",
        yaxis_title="Clasificación real",
        height=390,
        margin=dict(
            l=25, r=25, t=25, b=65
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        figura_matriz,
        use_container_width=True
    )

    st.caption(
        "Cada fila representa la categoría real "
        "y cada columna la categoría predicha "
        "por la red bayesiana."
    )

    st.markdown("#### Distribución del dataset")

    distribucion = sistema["distribucion"]

    figura_datos = go.Figure()

    figura_datos.add_trace(
        go.Bar(
            x=CLASES,
            y=[
                int(distribucion[c])
                for c in CLASES
            ],
            marker_color=[
                COLORES[c]
                for c in CLASES
            ],
            text=[
                int(distribucion[c])
                for c in CLASES
            ],
            textposition="outside"
        )
    )

    figura_datos.update_layout(
        xaxis_title="Nivel de riesgo",
        yaxis_title="Cantidad de registros",
        height=330,
        margin=dict(
            l=20, r=20, t=20, b=35
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        figura_datos,
        use_container_width=True
    )

    with st.expander(
        "Ver muestra de datos académicos"
    ):

        st.dataframe(
            sistema["original"].head(10),
            hide_index=True,
            use_container_width=True
        )


# ============================================
# TAB 3 - METODOLOGIA
# ============================================

with tab3:

    st.markdown(
        '<div class="section-label">Módulo 03</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Fundamento científico del sistema'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
### Modelo inteligente

El sistema utiliza una **red bayesiana discreta
con arquitectura Naive Bayes**.

La estructura considera una variable objetivo,
siete variables predictoras y relaciones de
dependencia condicional.

### Procedimiento de entrenamiento

- Dataset académico: registros disponibles.
- Partición: 80 % entrenamiento y 20 % validación.
- Muestreo estratificado por nivel de riesgo.
- Semilla aleatoria: 42.
- Estimación bayesiana de parámetros.
- Suavizado de probabilidades K2.
- Inferencia por eliminación de variables.

### Variables académicas

1. Promedio académico.
2. Asistencia a clases.
3. Cursos desaprobados.
4. Horas semanales de estudio.
5. Hábitos de estudio.
6. Nivel de estrés académico.
7. Participación académica.

### Evaluación

El desempeño se examina mediante exactitud,
precisión macro, recall macro, F1 macro y
matriz de confusión.

Las métricas describen la capacidad de reproducir
las etiquetas del conjunto evaluado. Su validez
como predicción del riesgo real depende de la
procedencia y calidad del etiquetado.
    """)

    st.info(
        "La herramienta constituye un prototipo "
        "académico de apoyo a decisiones. "
        "No reemplaza los procedimientos "
        "institucionales de evaluación."
    )


# ============================================
# PIE DE PAGINA
# ============================================

st.markdown("""
<div class="footer">
    SIRA · Sistema Inteligente de Riesgo Académico<br>
    Universidad Nacional de Trujillo ·
    Inteligencia Artificial II · 2026
</div>
""", unsafe_allow_html=True)
