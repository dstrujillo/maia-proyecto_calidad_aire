# Análisis del Notebook: Exploración y Preprocesamiento de Datos

## Descripción General
Este notebook tiene como objetivo crear un pipeline de exploración, análisis y preparación de datos para un modelo que prediga el nivel de riesgo (bajo, medio, alto) basado en la ubicación geográfica e indicadores ambientales, con recomendaciones de actividades y precauciones.
Como salida se generan archivos con diferentes características que permiten ser adicionados a pipeline de modelado.

## Estructura del Notebook

Toda la explicación numérica está en el notebook /script_exploracion_preprocesamiento/proyecto.ipynb

### 1. Importación de Librerías
Se importan las bibliotecas necesarias para el análisis, incluyendo:
- `numpy` y `pandas` para manipulación de datos
- `matplotlib` y `seaborn` para visualización
- `scikit-learn` para construcción de modelos y métricas
- `scipy.stats` para análisis estadístico

### 2. Exploración de Datos
Se carga el dataset `air_quality_health_dataset.csv` y se realiza un análisis exploratorio inicial:

**Hallazgos principales:**
- El dataset contiene 88,489 registros con 12 columnas
- No hay valores nulos ni duplicados
- Se detectaron valores atípicos en varias columnas:
  - pm2_5: 301 valores atípicos (0.34%)
  - pm10: 340 valores atípicos (0.38%)
  - no2: 579 valores atípicos (0.65%)
  - o3: 616 valores atípicos (0.70%)
  - hospital_admissions: 1373 valores atípicos (1.55%)

### 3. Transformación de Datos
Se aplica un pipeline de preprocesamiento que genera tres conjuntos de características diferentes:

**Opción 1:** 15 características (incluyendo todas las variables numéricas originales)
**Opción 2:** 14 características (eliminando alguna variable)
**Opción 3:** 19 características (agregando características generadas)

### 4. Evaluación de los Datasets
Se analizan las correlaciones con la variable objetivo `hospital_admissions` para cada conjunto de características.

**Resultados del análisis de correlación:**
- Las características ambientales (aqi, pm2_5, pm10, no2, o3) muestran correlaciones significativas con hospital_admissions
- Las características generadas (interacciones y transformaciones) mejoran las correlaciones en algunos casos
- La temperatura y humedad también muestran correlaciones moderadas

## Selección de Características para la Etapa de Entrenamiento

### Características con Mayor Correlación
Basado en el análisis de correlación, se recomienda mantener las siguientes características para el modelo:

1. **aqi** (Índice de Calidad del Aire) - Alta correlación
2. **pm2_5** (Material particulado 2.5) - Alta correlación  
3. **pm10** (Material particulado 10) - Alta correlación
4. **no2** (Dióxido de Nitrógeno) - Correlación significativa
5. **o3** (Ozono) - Correlación significativa
6. **temperature** - Correlación moderada
7. **humidity** - Correlación moderada
8. **hospital_capacity** - Correlación inversa moderada

### Características Categóricas a Considerar
- **population_density** (Rural/Urban/Suburban) - Debe ser codificada adecuadamente

### Características Generadas Recomendadas
Del conjunto de características 3 (19 características), se recomiendan las transformaciones que mostraron mayor correlación con la variable objetivo, particularmente:
- Interacciones entre variables ambientales
- Transformaciones polinómicas de las variables más correlacionadas

## Recomendaciones para la Etapa de Entrenamiento

1. **Conjunto de características recomendado:** Opción 3 (19 características) ya que incluye transformaciones que mejoran las correlaciones
2. **Preprocesamiento necesario:**
   - Codificación one-hot para `population_density`
   - Estandarización de variables numéricas
   - Considerar tratamiento específico para valores atípicos en variables clave
3. **Variables objetivo:** Utilizar `hospital_admissions` como variable continua o categorizarla para clasificación de riesgo

## Archivos Generados
El notebook genera tres archivos preprocesados:
- `preprocessed_air_quality_data_feat_set1.csv`
- `preprocessed_air_quality_data_feat_set2.csv` 
- `preprocessed_air_quality_data_feat_set3.csv`

Estos archivos están listos para la etapa de modelado y entrenamiento.