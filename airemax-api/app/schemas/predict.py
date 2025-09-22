from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


AQI_DATA_SAMPLE_RESPONSE = {
        "aqi": 38,
        "idx": 6235,
        "attributions": [
            {
                "url": "http://oab.ambientebogota.gov.co/",
                "name": "OAB - El Observatorio Ambiental de Bogot&aacute;",
                "logo": "Colombia-OAB.png"
            },
            {
                "url": "https://waqi.info/",
                "name": "World Air Quality Index Project"
            }
        ],
        "city": {
            "geo": [
                4.57619,
                -74.13093
            ],
            "name": "Tunal, Bogota, Colombia",
            "url": "https://aqicn.org/city/colombia/bogota/tunal",
            "location": ""
        },
        "dominentpol": "pm25",
        "iaqi": {
            "dew": {
                "v": 12
            },
            "h": {
                "v": 67
            },
            "no2": {
                "v": 3
            },
            "o3": {
                "v": 7.6
            },
            "p": {
                "v": 1029
            },
            "pm10": {
                "v": 18
            },
            "pm25": {
                "v": 38
            },
            "r": {
                "v": 0.1
            },
            "so2": {
                "v": 0.1
            },
            "t": {
                "v": 15
            },
            "w": {
                "v": 0.9
            },
            "wd": {
                "v": 179
            },
            "wg": {
                "v": 9.7
            }
        },
        "time": {
            "s": "2025-09-18 09:00:00",
            "tz": "-05:00",
            "v": 1758186000,
            "iso": "2025-09-18T09:00:00-05:00"
        },
        "forecast": {
            "daily": {
                "pm10": [
                    {
                        "avg": 11,
                        "day": "2025-09-16",
                        "max": 16,
                        "min": 6
                    },
                    {
                        "avg": 15,
                        "day": "2025-09-17",
                        "max": 23,
                        "min": 11
                    },
                    {
                        "avg": 15,
                        "day": "2025-09-18",
                        "max": 21,
                        "min": 9
                    },
                    {
                        "avg": 15,
                        "day": "2025-09-19",
                        "max": 24,
                        "min": 6
                    },
                    {
                        "avg": 14,
                        "day": "2025-09-20",
                        "max": 27,
                        "min": 5
                    },
                    {
                        "avg": 15,
                        "day": "2025-09-21",
                        "max": 28,
                        "min": 8
                    },
                    {
                        "avg": 20,
                        "day": "2025-09-22",
                        "max": 39,
                        "min": 10
                    }
                ],
                "pm25": [
                    {
                        "avg": 35,
                        "day": "2025-09-16",
                        "max": 55,
                        "min": 19
                    },
                    {
                        "avg": 50,
                        "day": "2025-09-17",
                        "max": 66,
                        "min": 34
                    },
                    {
                        "avg": 50,
                        "day": "2025-09-18",
                        "max": 65,
                        "min": 34
                    },
                    {
                        "avg": 51,
                        "day": "2025-09-19",
                        "max": 72,
                        "min": 20
                    },
                    {
                        "avg": 45,
                        "day": "2025-09-20",
                        "max": 80,
                        "min": 17
                    },
                    {
                        "avg": 47,
                        "day": "2025-09-21",
                        "max": 82,
                        "min": 29
                    },
                    {
                        "avg": 60,
                        "day": "2025-09-22",
                        "max": 101,
                        "min": 35
                    }
                ],
                "uvi": [
                    {
                        "avg": 0,
                        "day": "2025-09-17",
                        "max": 0,
                        "min": 0
                    },
                    {
                        "avg": 2,
                        "day": "2025-09-18",
                        "max": 8,
                        "min": 0
                    },
                    {
                        "avg": 2,
                        "day": "2025-09-19",
                        "max": 8,
                        "min": 0
                    },
                    {
                        "avg": 2,
                        "day": "2025-09-20",
                        "max": 10,
                        "min": 0
                    },
                    {
                        "avg": 3,
                        "day": "2025-09-21",
                        "max": 12,
                        "min": 0
                    },
                    {
                        "avg": 4,
                        "day": "2025-09-22",
                        "max": 11,
                        "min": 0
                    }
                ]
            }
        },
        "debug": {
            "sync": "2025-09-18T23:08:33+09:00"
        }
    }

class AQIHealthAnalysis(BaseModel):
    level: str = Field(..., description="Nivel de calidad del aire")
    color_code: str = Field(..., description="Código de color hexadecimal asociado al nivel")
    health_implications: str = Field(..., description="Descripción del nivel de calidad del aire")
    cautionary_statement: str = Field(..., description="Recomendaciones de precaución para la salud")

class PollutantLevel(BaseModel):
    description: str = Field(..., description="Descripción del nivel de calidad del aire")
    color: str = Field(..., description="Código de color hexadecimal asociado al nivel")

class PollutantData(BaseModel):
    value: Optional[float] = Field(None, description="Valor numérico del contaminante")
    units: Optional[str] = Field(None, description="Unidades de medida")
    level: Optional[str] = Field(None, description="Descripción del nivel de calidad del aire")
    color: Optional[str] = Field(None, description="Código de color hexadecimal asociado al nivel")

class IAQIData(BaseModel):
    value: Optional[float] = Field(None, description="Valor numérico del indicador")
    units: Optional[str] = Field(None, description="Unidades de medida")
    name: Optional[str] = Field(None, description="Nombre del indicador")


class ForecastItem(BaseModel):
    day: str = Field(..., description="Día de la predicción (YYYY-MM-DD)", example="2025-09-16")
    avg: float = Field(..., description="Valor promedio del contaminante", example=11.0)
    min: float = Field(..., description="Valor mínimo del contaminante", example=6.0)
    max: float = Field(..., description="Valor máximo del contaminante", example=16.0)

class AirQualityResponse(BaseModel):
    # Indicador Principal ICA
    #aqi_data: Dict[str, Any] = Field(..., description="Diccionario datos AQI", example=2)
    aqi : int = Field(..., description="Valor numérico del AQI", example=38)
    aqi_health_analysis: AQIHealthAnalysis = Field(..., description="Análisis de salud basado en AQI")
    risk_prediction: str = Field(..., description="Predicción de riesgo en próximas horas")
    forecast: Dict[str, List[ForecastItem]] = Field(..., description="Predicciones de contaminantes")
    
    # Monitoreo en Tiempo Real - Contaminantes
    pollutants: Dict[str, PollutantData] = Field(..., description="Datos de contaminantes")
    
    # Variables
    iaqi: Dict[str, IAQIData] = Field(..., description="Datos base de indicadores meteorológicos")
    

    # Información de ubicación
    location: Dict[str, Any] = Field(..., description="Información de la ubicación consultada")
    
    class Config:
        schema_extra = {
            "example": {
                "aqi": 13,
                "aqi_health_analysis": {
                    "level": "Bueno",
                    "color_code": "#4CAF50",
                    "health_implications": "Sin riesgos para la salud",
                    "cautionary_statement": "Ninguna"
                },
                "risk_prediction": "La calidad del aire se mantendrá buena/moderada durante las próximas horas",
                "forecast": {
                    "pm10": [
                    {
                        "day": "2025-09-16",
                        "avg": 10,
                        "min": 10,
                        "max": 11
                    },
                    {
                        "day": "2025-09-17",
                        "avg": 16,
                        "min": 11,
                        "max": 23
                    },
                    {
                        "day": "2025-09-18",
                        "avg": 15,
                        "min": 9,
                        "max": 20
                    },
                    {
                        "day": "2025-09-19",
                        "avg": 17,
                        "min": 9,
                        "max": 33
                    },
                    {
                        "day": "2025-09-20",
                        "avg": 10,
                        "min": 5,
                        "max": 16
                    },
                    {
                        "day": "2025-09-21",
                        "avg": 12,
                        "min": 8,
                        "max": 15
                    },
                    {
                        "day": "2025-09-22",
                        "avg": 19,
                        "min": 9,
                        "max": 50
                    },
                    {
                        "day": "2025-09-23",
                        "avg": 15,
                        "min": 9,
                        "max": 28
                    }
                    ],
                    "pm25": [
                    {
                        "day": "2025-09-16",
                        "avg": 30,
                        "min": 29,
                        "max": 31
                    },
                    {
                        "day": "2025-09-17",
                        "avg": 52,
                        "min": 34,
                        "max": 66
                    },
                    {
                        "day": "2025-09-18",
                        "avg": 54,
                        "min": 37,
                        "max": 65
                    },
                    {
                        "day": "2025-09-19",
                        "avg": 54,
                        "min": 31,
                        "max": 88
                    },
                    {
                        "day": "2025-09-20",
                        "avg": 34,
                        "min": 16,
                        "max": 51
                    },
                    {
                        "day": "2025-09-21",
                        "avg": 41,
                        "min": 29,
                        "max": 52
                    },
                    {
                        "day": "2025-09-22",
                        "avg": 57,
                        "min": 30,
                        "max": 130
                    },
                    {
                        "day": "2025-09-23",
                        "avg": 47,
                        "min": 32,
                        "max": 80
                    }
                    ],
                    "uvi": [
                    {
                        "day": "2025-09-17",
                        "avg": 0,
                        "min": 0,
                        "max": 0
                    },
                    {
                        "day": "2025-09-18",
                        "avg": 2,
                        "min": 0,
                        "max": 8
                    },
                    {
                        "day": "2025-09-19",
                        "avg": 2,
                        "min": 0,
                        "max": 8
                    },
                    {
                        "day": "2025-09-20",
                        "avg": 2,
                        "min": 0,
                        "max": 10
                    },
                    {
                        "day": "2025-09-21",
                        "avg": 3,
                        "min": 0,
                        "max": 12
                    },
                    {
                        "day": "2025-09-22",
                        "avg": 4,
                        "min": 0,
                        "max": 11
                    }
                    ]
                },
                "pollutants": {
                    "PM2.5": {
                    "value": 13,
                    "units": "μg/m³",
                    "level": "Moderado",
                    "color": "#FFEB3B"
                    },
                    "PM10": {
                    "value": 10,
                    "units": "μg/m³",
                    "level": "Bueno",
                    "color": "#4CAF50"
                    },
                    "O₃": {
                    "value": 9.2,
                    "units": "μg/m³",
                    "level": "Bueno",
                    "color": "#4CAF50"
                    },
                    "NO₂": {
                    "value": 2.6,
                    "units": "μg/m³",
                    "level": "Bueno",
                    "color": "#4CAF50"
                    },
                    "SO₂": {
                    "value": 0.4,
                    "units": "μg/m³",
                    "level": "Bueno",
                    "color": "#4CAF50"
                    },
                    "CO": {
                    "value": 2.7,
                    "units": "mg/m³",
                    "level": "Bueno",
                    "color": "#4CAF50"
                    }
                },
                "iaqi": {
                    "CO": {
                    "value": 2.7,
                    "units": "mg/m³",
                    "name": "Monóxido de Carbono"
                    },
                    "Dew Point": {
                    "value": 11,
                    "units": "°C",
                    "name": "Punto de Rocío"
                    },
                    "Humedad": {
                    "value": 58,
                    "units": "%",
                    "name": "Humedad Relativa"
                    },
                    "NO₂": {
                    "value": 2.6,
                    "units": "μg/m³",
                    "name": "Dióxido de Nitrógeno"
                    },
                    "O₃": {
                    "value": 9.2,
                    "units": "μg/m³",
                    "name": "Ozono"
                    },
                    "Presión": {
                    "value": 1027,
                    "units": "hPa",
                    "name": "Presión Atmosférica"
                    },
                    "PM10": {
                    "value": 10,
                    "units": "μg/m³",
                    "name": "Material Particulado PM10"
                    },
                    "PM2.5": {
                    "value": 13,
                    "units": "μg/m³",
                    "name": "Material Particulado PM2.5"
                    },
                    "Lluvia": {
                    "value": 0.1,
                    "units": "mm",
                    "name": "Precipitación"
                    },
                    "SO₂": {
                    "value": 0.4,
                    "units": "μg/m³",
                    "name": "Dióxido de Azufre"
                    },
                    "Temperatura": {
                    "value": 16.4,
                    "units": "°C",
                    "name": "Temperatura"
                    },
                    "Viento": {
                    "value": 2,
                    "units": "m/s",
                    "name": "Velocidad del Viento"
                    },
                    "Dirección Viento": {
                    "value": 120,
                    "units": "°",
                    "name": "Dirección del Viento"
                    },
                    "Ráfagas": {
                    "value": 9.7,
                    "units": "m/s",
                    "name": "Ráfagas de Viento"
                    }
                },
                "location": {
                    "name": "San Cristobal, Bogota, Colombia",
                    "lat": 3.7128,
                    "lon": -74.006
                }
                }
        }