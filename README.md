
# SIRA - Sistema Inteligente de Riesgo Académico

**Universidad Nacional de Trujillo**

**Curso:** Inteligencia Artificial II

**Año:** 2026

## Descripción

Sistema inteligente basado en conocimiento
mediante redes bayesianas para evaluar el
riesgo académico estudiantil.

El sistema permite procesar información
académica y estimar probabilidades de
riesgo bajo, medio y alto.

## Objetivo general

Implementar un sistema inteligente basado
en conocimiento mediante redes bayesianas
para evaluar el riesgo académico estudiantil,
Trujillo, 2026.

## Tecnologías

- Python
- pgmpy
- Scikit-learn
- Pandas
- Plotly
- Streamlit Community Cloud
- GitHub

## Arquitectura del modelo

Modelo: Naive Bayes discreto.

Configuración:

- 8 nodos.
- 7 relaciones dirigidas.
- 7 variables predictoras.
- 3 clases de riesgo académico.
- Estimador bayesiano K2.
- Inferencia mediante eliminación de variables.

## Preparación y entrenamiento

El sistema procesa el conjunto de datos
académicos y aplica una partición
estratificada de 80 % para entrenamiento
y 20 % para validación.

El modelo se ajusta con el subconjunto
de entrenamiento y genera predicciones
sobre los registros reservados para validación.

## Evaluación

Se calculan los siguientes indicadores:

- Exactitud.
- Precisión macro.
- Recall macro.
- F1 macro.
- Matriz de confusión.

## Interfaz web

La aplicación permite:

1. Registrar características académicas.
2. Ejecutar inferencia probabilística.
3. Mostrar una clasificación de riesgo.
4. Visualizar las probabilidades por categoría.
5. Consultar indicadores técnicos del modelo.

## Estructura del proyecto

- app.py: interfaz de usuario.
- modelo.py: procesamiento, entrenamiento e inferencia.
- requirements.txt: dependencias.
- datos/estudiantes.csv: dataset preparado.
- .streamlit/config.toml: identidad visual.

## Aplicación publicada

Agregar aquí el enlace real de Streamlit
después del despliegue.

## Limitaciones

Este proyecto constituye un prototipo académico.

Las métricas describen el desempeño frente
a las etiquetas disponibles en el conjunto
de datos. Su interpretación depende de
la procedencia y validez del etiquetado.

El sistema no reemplaza la evaluación
académica realizada por profesionales.

## Autores

- Miguel Ángel Chuqui Cueva.
- Henry Martin Rodríguez Vásquez.
- Jhonny Jesús Zavaleta Gadea.
