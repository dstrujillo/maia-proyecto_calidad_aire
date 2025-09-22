import json
from typing import Any
import pprint

#import numpy as np
#import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from loguru import logger
from model import __version__ as model_version
#from model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

from app.data.aiqcn_api_customer import get_air_quality_data

api_router = APIRouter()

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

@api_router.get("/air-quality", response_model=schemas.AirQualityResponse, status_code=200)
async def get_air_quality(
    lat: float = Query(..., description="Latitud de la ubicación", example=40.7128),
    lon: float = Query(..., description="Longitud de la ubicación", example=-74.0060)
) -> Any:
    """
    Obtiene datos de calidad del aire y meteorológicos para una ubicación específica.
    Basado en los datos, genera predicción y recomendación para reducción del riesgo de enfermedades y accidentes respiratorios.
    
    Args:
        lat (float): Latitud de la ubicación
        lon (float): Longitud de la ubicación
    
    Returns:
        Objeto JSON con:
        - AQI y análisis de salud
        - Predicción de riesgo
        - Pronósticos de contaminantes
        - Datos de contaminantes en tiempo real
    Note:
    
    Esta función utiliza la API de AQICN (https://aqicn.org/json-api/doc/) para obtener los datos de estaciones meteorológicas más cercanas a las coordenadas proporcionadas.
    """
    try:
        logger.info(f"Fetching air quality data for lat: {lat}, lon: {lon}")
        
        # Obtener datos de la API externa
        results = get_air_quality_data(lat, lon)

        pprint.pprint(results, indent=4, width=80, depth=None)
        
        if results.get("errors"):
            logger.error(f"Error fetching air quality data: {results.get('errors')}")
            raise HTTPException(status_code=500, detail=results.get("errors"))
        
        logger.info(f"Air quality data retrieved successfully")
        return results
        
    except Exception as e:
        logger.error(f"Unexpected error in get_air_quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")
