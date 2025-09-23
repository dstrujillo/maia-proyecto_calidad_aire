# Microproyecto – Modelo Predictivo de Calidad del Aire y Alertas de Salud  

# Proyecto Calidad del Aire

Este proyecto busca predecir la calidad del aire usando técnicas de Machine Learning (ML), **DVC** y **AWS S3**, garantizando reproducibilidad, versionamiento de datos y trazabilidad de resultados.

---

# Infraestructura de Datos

La infraestructura del proyecto integra **DVC** con un bucket en **AWS S3**, lo que permite versionar datasets y sincronizarlos en la nube.  
A continuación, se presenta la **estructura de alto nivel del proyecto** y el **flujo de datos** que sigue nuestro pipeline.

---

##  Estructura del Proyecto

```plaintext
maia-proyecto_calidad_aire/
│
├── airemax-api/                          # API para exponer el modelo
├── data/                                 # Datos versionados con DVC
│   ├── raw/                              # Datos crudos
│   ├── processed/                        # Datos procesados
│   └── results/                          # Resultados intermedios
├── entrega2/                             # Entregables de la segunda fase
├── evidences/                            # Capturas y documentación Entrega 3
├── mlruns/                               # Tracking de experimentos con MLflow
├── results/                              # Resultados finales del modelo
├── script_exploracion_preprocesamiento/  # Exploración y limpieza de datos
├── scripts_entrenamientos/               # Scripts de entrenamiento de modelos
├── visualizations/                       # Visualizaciones y gráficas
│
├── .dvc/                                 # Configuración interna de DVC
│   └── config                            # Configuración del remote S3
├── .dvcignore                            # Archivos ignorados por DVC
├── README.md                             # Documentación principal
├── requirements.txt                      # Dependencias del proyecto
├── dvc.yaml / dvc.lock                   # Definición del pipeline de datos
└── data/raw/test.csv.dvc                 # Ejemplo de dataset agregado a DVC
📊 Flujo de Datos
El pipeline sigue las siguientes etapas:

flowchart TD
    A[📥 Dataset crudo en S3] -->|dvc pull| B[data/raw]
    B --> C[🧹 Preprocesamiento]
    C --> D[⚙️ Feature Engineering]
    D --> E[🤖 Entrenamiento del modelo]
    E --> F[📊 Evaluación de métricas]
    F --> G[📁 Resultados locales y en S3]
yaml


## 📊 Flujo de Datos y Procesos

El siguiente diagrama muestra el flujo completo desde los datos crudos hasta los resultados:

### 🖼️ Versión Renderizada (Imagen)
![Flujo de Datos](docs/flujo_datos_completo.png)

### 📐 Versión en Mermaid (Reproducible)
```mermaid
flowchart TD
    A[Dataset crudo en S3] -->|dvc pull| B[data/raw]
    B --> C[Preprocesamiento]
    C --> D[Feature Engineering]
    D --> E[Entrenamiento del modelo]
    E --> F[Evaluación de métricas]
    F --> G[Resultados y reportes]



## 📊 Evidencia AWS S3

El siguiente pantallazo muestra el listado del bucket en AWS S3, confirmando que la infraestructura de datos está configurada correctamente:

![Evidencia Bucket S3](docs/aws_s3_bucket_list.png)



##  Evidencias de DVC + AWS S3

El proyecto utiliza **DVC** conectado a un bucket en **AWS S3** para versionar datos y resultados.  
A continuación, se muestran ejemplos de ejecución de los principales comandos:

### 🔹 Estado del pipeline
```bash
$ dvc status
Data and pipelines are up to date.



## 📊 Resultados Versionados con DVC

Las métricas de entrenamiento (`training_results.csv`) están versionadas con **DVC** y respaldadas en nuestro bucket de **AWS S3**.  

Ejemplo de verificación:

```bash
dvc status
# Data and pipelines are up to date.
Puedes recuperar las métricas desde S3 en cualquier momento con:

bash
Copy code
dvc pull results/training_results.csv
📂 Evidencias
Archivo versionado: results/training_results.csv

Respaldo en S3: s3://maia-calidad-aire-mackie

Estado actual: ✅ sincronizado y actualizado

yaml


---

## 🖼️ Evidencias Visuales

Además de los archivos versionados con DVC, incluimos pantallazos como respaldo visual del estado del proyecto.

### Evidencia 1 – Bucket en AWS S3
![Evidencia Bucket](docs/aws_s3_bucket_list.png)

### Evidencia 2 – DVC sincronizado
Ejemplo de comando ejecutado:

```bash
dvc push
# Everything is up to date.
Evidencia 3 – Estado del pipeline
bash
Copy code
dvc status
# Data and pipelines are up to date.
yaml
Copy code

---

