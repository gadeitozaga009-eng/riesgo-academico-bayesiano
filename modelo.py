
from pathlib import Path

import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================
# CONFIGURACION DEL MODELO
# ============================================

BASE = Path(__file__).resolve().parent
RUTA_DATOS = BASE / "datos" / "estudiantes.csv"

CLASES = ["Bajo", "Medio", "Alto"]

VARIABLES = [
    "Promedio",
    "Asistencia",
    "Desaprobados",
    "Horas",
    "Habitos",
    "Estres",
    "Participacion"
]

ESTADOS = {
    "Riesgo": CLASES,
    "Promedio": ["Bajo", "Medio", "Alto"],
    "Asistencia": ["Baja", "Media", "Alta"],
    "Desaprobados": [
        "Ninguno", "Uno", "Dos_o_mas"
    ],
    "Horas": ["Baja", "Media", "Alta"],
    "Habitos": [
        "Adecuado", "Regular", "Inadecuado"
    ],
    "Estres": ["Bajo", "Moderado", "Alto"],
    "Participacion": [
        "Baja", "Media", "Alta"
    ]
}

COLUMNAS = [
    "Promedio_Obtenido",
    "Asistencia_Porcentaje",
    "Cursos_Desaprobados",
    "Horas_Estudio_Semanales",
    "Habitos_Estudio",
    "Nivel_Estres_Academico",
    "Participacion_Academica",
    "Nivel_Riesgo_Academico"
]


# ============================================
# PREPROCESAMIENTO
# ============================================

def convertir_numero(serie):
    return pd.to_numeric(
        serie.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


def preparar_entrada(
    promedio,
    asistencia,
    desaprobados,
    horas,
    habitos,
    estres,
    participacion
):
    if promedio < 11:
        categoria_promedio = "Bajo"
    elif promedio < 15:
        categoria_promedio = "Medio"
    else:
        categoria_promedio = "Alto"

    if asistencia < 75:
        categoria_asistencia = "Baja"
    elif asistencia < 90:
        categoria_asistencia = "Media"
    else:
        categoria_asistencia = "Alta"

    if desaprobados == 0:
        categoria_desaprobados = "Ninguno"
    elif desaprobados == 1:
        categoria_desaprobados = "Uno"
    else:
        categoria_desaprobados = "Dos_o_mas"

    if horas < 8:
        categoria_horas = "Baja"
    elif horas < 15:
        categoria_horas = "Media"
    else:
        categoria_horas = "Alta"

    return {
        "Promedio": categoria_promedio,
        "Asistencia": categoria_asistencia,
        "Desaprobados": categoria_desaprobados,
        "Horas": categoria_horas,
        "Habitos": habitos,
        "Estres": estres,
        "Participacion": participacion
    }


def cargar_dataset():

    # Detecta separadores de CSV exportados
    # desde diferentes configuraciones de Excel.
    df = pd.read_csv(
        RUTA_DATOS,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    df.columns = df.columns.str.strip()

    faltantes = [
        col for col in COLUMNAS
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas: " + ", ".join(faltantes)
        )

    df = df[COLUMNAS].copy()

    numericas = [
        "Promedio_Obtenido",
        "Asistencia_Porcentaje",
        "Cursos_Desaprobados",
        "Horas_Estudio_Semanales"
    ]

    for col in numericas:
        df[col] = convertir_numero(df[col])

    categoricas = [
        "Habitos_Estudio",
        "Nivel_Estres_Academico",
        "Participacion_Academica",
        "Nivel_Riesgo_Academico"
    ]

    for col in categoricas:
        df[col] = (
            df[col].astype("string")
            .str.strip()
            .str.title()
        )

    if df.isna().any().any():
        raise ValueError(
            "Existen valores vacios o invalidos "
            "en el dataset."
        )

    if not df["Promedio_Obtenido"].between(
        0, 20
    ).all():
        raise ValueError(
            "Se encontraron promedios fuera de rango."
        )

    if not df["Asistencia_Porcentaje"].between(
        0, 100
    ).all():
        raise ValueError(
            "Se encontraron asistencias fuera de rango."
        )

    if (
        (df["Cursos_Desaprobados"] < 0).any()
        or (df["Horas_Estudio_Semanales"] < 0).any()
    ):
        raise ValueError(
            "Se encontraron valores negativos invalidos."
        )

    registros = []

    for _, fila in df.iterrows():

        registro = preparar_entrada(
            fila["Promedio_Obtenido"],
            fila["Asistencia_Porcentaje"],
            fila["Cursos_Desaprobados"],
            fila["Horas_Estudio_Semanales"],
            fila["Habitos_Estudio"],
            fila["Nivel_Estres_Academico"],
            fila["Participacion_Academica"]
        )

        registro["Riesgo"] = (
            fila["Nivel_Riesgo_Academico"]
        )

        registros.append(registro)

    datos = pd.DataFrame(
        registros,
        columns=VARIABLES + ["Riesgo"]
    )

    # Verificar los estados de las variables
    for variable, valores_validos in ESTADOS.items():

        encontrados = set(datos[variable])

        invalidos = (
            encontrados - set(valores_validos)
        )

        if invalidos:
            raise ValueError(
                f"Valores invalidos en {variable}: "
                f"{invalidos}"
            )

    return df, datos



# ============================================
# ENTRENAMIENTO BAYESIANO CORREGIDO
# ============================================

def entrenar_modelo(datos_entrenamiento):

    # 1. Definir las relaciones de la red
    relaciones = [
        ("Riesgo", variable)
        for variable in VARIABLES
    ]

    # 2. Construir la red bayesiana discreta
    red = DiscreteBayesianNetwork(relaciones)

    # 3. Inicializar el estimador bayesiano
    estimador = BayesianEstimator(
        model=red,
        data=datos_entrenamiento,
        state_names=ESTADOS
    )

    # 4. Calcular las probabilidades con K2
    probabilidades = estimador.get_parameters(
        prior_type="K2"
    )

    # 5. Incorporar las probabilidades a la red
    red.add_cpds(*probabilidades)

    # 6. Verificar la validez del modelo
    if not red.check_model():
        raise ValueError(
            "La red bayesiana no es valida."
        )

    # 7. Devolver la red entrenada
    return red


# ============================================
# INFERENCIA
# ============================================

def inferir(motor, evidencia):

    resultado = motor.query(
        variables=["Riesgo"],
        evidence=evidencia,
        show_progress=False
    )

    probabilidades = {
        estado: float(valor)
        for estado, valor in zip(
            resultado.state_names["Riesgo"],
            resultado.values
        )
    }

    clasificacion = max(
        probabilidades,
        key=probabilidades.get
    )

    return clasificacion, probabilidades


def predecir_conjunto(motor, datos):

    predicciones = []

    for _, fila in datos.iterrows():

        evidencia = {
            variable: fila[variable]
            for variable in VARIABLES
        }

        clasificacion, _ = inferir(
            motor, evidencia
        )

        predicciones.append(clasificacion)

    return predicciones


# ============================================
# METRICAS
# ============================================

def calcular_metricas(reales, predicciones):

    return {
        "Exactitud": accuracy_score(
            reales, predicciones
        ),
        "Precision macro": precision_score(
            reales,
            predicciones,
            labels=CLASES,
            average="macro",
            zero_division=0
        ),
        "Recall macro": recall_score(
            reales,
            predicciones,
            labels=CLASES,
            average="macro",
            zero_division=0
        ),
        "F1 macro": f1_score(
            reales,
            predicciones,
            labels=CLASES,
            average="macro",
            zero_division=0
        )
    }


# ============================================
# EJECUCION DEL EXPERIMENTO
# ============================================

def construir_sistema():

    datos_originales, datos = cargar_dataset()

    entrenamiento, validacion = train_test_split(
        datos,
        test_size=0.20,
        random_state=42,
        stratify=datos["Riesgo"]
    )

    red = entrenar_modelo(entrenamiento)

    motor = VariableElimination(red)

    pred_train = predecir_conjunto(
        motor, entrenamiento
    )

    pred_val = predecir_conjunto(
        motor, validacion
    )

    metricas_train = calcular_metricas(
        entrenamiento["Riesgo"],
        pred_train
    )

    metricas_val = calcular_metricas(
        validacion["Riesgo"],
        pred_val
    )

    matriz = confusion_matrix(
        validacion["Riesgo"],
        pred_val,
        labels=CLASES
    )

    distribucion = (
        datos["Riesgo"]
        .value_counts()
        .reindex(CLASES, fill_value=0)
    )

    return {
        "modelo": red,
        "motor": motor,
        "original": datos_originales,
        "datos": datos,
        "n_train": len(entrenamiento),
        "n_val": len(validacion),
        "metricas_train": metricas_train,
        "metricas_val": metricas_val,
        "matriz": matriz,
        "distribucion": distribucion
    }
