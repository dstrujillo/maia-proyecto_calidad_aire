import requests
from loguru import logger
from typing import Dict, Any
from app.config import settings

# Diccionario de mapeo de indicadores
AQICN_INDICATORS = {
    "co": {"name": "Monóxido de Carbono", "units": "mg/m³", "standard": "CO"},
    "no2": {"name": "Dióxido de Nitrógeno", "units": "μg/m³", "standard": "NO₂"},
    "o3": {"name": "Ozono", "units": "μg/m³", "standard": "O₃"},
    "pm10": {"name": "Material Particulado PM10", "units": "μg/m³", "standard": "PM10"},
    "pm25": {"name": "Material Particulado PM2.5", "units": "μg/m³", "standard": "PM2.5"},
    "so2": {"name": "Dióxido de Azufre", "units": "μg/m³", "standard": "SO₂"},
    "dew": {"name": "Punto de Rocío", "units": "°C", "standard": "Dew Point"},
    "h": {"name": "Humedad Relativa", "units": "%", "standard": "Humedad"},
    "p": {"name": "Presión Atmosférica", "units": "hPa", "standard": "Presión"},
    "t": {"name": "Temperatura", "units": "°C", "standard": "Temperatura"},
    "w": {"name": "Velocidad del Viento", "units": "m/s", "standard": "Viento"},
    "wd": {"name": "Dirección del Viento", "units": "°", "standard": "Dirección Viento"},
    "wg": {"name": "Ráfagas de Viento", "units": "m/s", "standard": "Ráfagas"},
    "r": {"name": "Precipitación", "units": "mm", "standard": "Lluvia"}
}

def get_air_quality_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Obtiene datos de calidad del aire desde la API de AQICN.
    
    Args:
        lat (float): Latitud de la ubicación
        lon (float): Longitud de la ubicación
    
    Returns:
        Dict: Diccionario con los datos de calidad del aire procesados
    """
    try:
        # URL de la API de AQICN (necesitarás obtener un token)
        url = f"{settings.AQICN_API_URL}{lat};{lon}/?token={settings.AQICN_TOKEN}"
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] != "ok":
            return {"errors": "No se pudieron obtener los datos de calidad del aire"}
        
        # Procesar los datos de la API externa
        data['lat'] = lat
        data['lon'] = lon
        return process_aqicn_data(data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to AQICN API: {str(e)}")
        return {"errors": f"Error al conectar con el servicio de calidad del aire: {str(e)}"}
    except Exception as e:
        logger.error(f"Error processing AQICN data: {str(e)}")
        return {"errors": f"Error al procesar los datos de calidad del aire: {str(e)}"}

def process_aqicn_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesa los datos crudos de la API de AQICN al formato esperado por nuestra API.
    
    Args:
        data (Dict): Datos crudos de la API de AQICN
    
    Returns:
        Dict: Datos procesados en nuestro formato estándar
    """
    # Implementar la lógica de procesamiento según la estructura de datos de AQICN
    # Esta es una implementación de ejemplo
    aqi = data.get("data", {}).get("aqi", 0)
    
    # Determinar nivel y color basado en el AQI
    aqi_health_analysis = get_aqi_level_and_color(aqi)
    
    # Procesar contaminantes
    pollutants = {}
    iaqi = data.get("data", {}).get("iaqi", {})

    for pollutant_key in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
        if pollutant_key in iaqi:
            pollutant_value = iaqi[pollutant_key].get("v")
            pollutant_info = get_pollutant_level(pollutant_key, pollutant_value)
            pollutants[AQICN_INDICATORS[pollutant_key]["standard"]] = {
                "value": pollutant_value,
                "units": AQICN_INDICATORS[pollutant_key]["units"],
                "level": pollutant_info["description"],
                "color": pollutant_info["color"]
            }
    
    # Procesar otros contaminantes de manera similar...
    
    # Procesar datos meteorológicos
    # Complementar para mejor legibilidad
    prettier_iaqi = {
    }

    for key, value in iaqi.items():
        if key in AQICN_INDICATORS:
            prettier_iaqi[AQICN_INDICATORS[key]["standard"]] = {
                "value": value.get("v"),
                "units": AQICN_INDICATORS[key]["units"],
                "name": AQICN_INDICATORS[key]["name"]
        }

    
    # Generar URLs de gráficos (esto dependerá de tu implementación)
    #charts = {
    #    "pm25_trend": f"https://your-api.com/charts/pm25?lat={lat}&lon={lon}",
    #    "temperature_trend": f"https://your-api.com/charts/temperature?lat={lat}&lon={lon}"
    #}
    
    # Generar predicción de riesgo
    risk_prediction = generate_risk_prediction(aqi, data)

    # Información de ubicación
    # Estación metereológica más cercana
    location = {
        "name": data.get("data", {}).get("city", {}).get("name", ""),
        "lat": data.get("lat", data.get("data", {}).get("city", {}).get("geo", [0, 0])[0]),
        "lon": data.get("lon", data.get("data", {}).get("city", {}).get("geo", [0, 0])[1]),
        #"city": data.get("data", {}).get("city", {}).get("name", "").split(",")[0] if data.get("data", {}).get("city", {}).get("name") else "",
        #"country": data.get("data", {}).get("city", {}).get("name", "").split(",")[-1].strip() if data.get("data", {}).get("city", {}).get("name") else ""
    }
    
    return {
        #"aqi_data": data,
        "aqi": aqi,
        "aqi_health_analysis": aqi_health_analysis,
        "forecast": data.get("data", {}).get("forecast", {}).get("daily", {}),
        "risk_prediction": risk_prediction,
        "pollutants": pollutants,
        "iaqi": prettier_iaqi,
        "location": location
    }

def get_aqi_level_and_color(aqi: int) -> tuple:
    """Determina el nivel y color basado en el valor AQI."""
    statements = {}
    if aqi <= 50:
        statements = {
            "level" : "Bueno",
            "health_implications" : "Sin riesgos para la salud",
            "cautionary_statement" : "Ninguna",
            "color_code" : "#4CAF50"
        }
    elif aqi <= 100:
        statements = {
            "level" : "Moderado",
            "health_implications" : "Calidad del aire aceptable; sin embargo, para algunos contaminantes puede haber un riesgo moderado para la salud de un número muy pequeño de personas que son inusualmente sensibles a la contaminación del aire.",
            "cautionary_statement" : "Las personas que son inusualmente sensibles a la contaminación del aire deberían considerar limitar el tiempo al aire libre.",
            "color_code" : "#FFEB3B"
        }
    elif aqi <= 150:
        statements = {
            "level" : "Poco saludable para grupos sensibles",
            "health_implications" : "La calidad del aire es aceptable; sin embargo, puede haber un riesgo moderado para la salud de un número muy pequeño de personas que son inusualmente sensibles a la contaminación del aire.",
            "cautionary_statement" : "Las personas que son inusualmente sensibles a la contaminación del aire deberían considerar limitar el tiempo al aire libre.",
            "color_code" : "#FF9800"
        } 
    elif aqi <= 200:
        statements = {
            "level" : "Insalubre",
            "health_implications" : "La calidad del aire puede ser insalubre para las personas sensibles. Los miembros de grupos sensibles pueden experimentar efectos para la salud. El público en general probablemente no se verá afectado.",
            "cautionary_statement" : "Los miembros de grupos sensibles deben limitar el tiempo al aire libre.",
            "color_code" : "#F44336"
        }
    elif aqi <= 300:
        statements = {
            "level" : "Muy insalubre",
            "health_implications" : "La calidad del aire es insalubre. Todos pueden comenzar a experimentar efectos para la salud; los miembros de grupos sensibles pueden experimentar efectos más graves para la salud.",
            "cautionary_statement" : "Todos deben limitar el tiempo al aire libre; los miembros de grupos sensibles deben evitar el tiempo al aire libre.",
            "color_code" : "#9C27B0"
        }
    else:
        statements = {
            "level" : "Peligroso",
            "health_implications" : "La calidad del aire es muy insalubre. Advertencias de salud de condiciones de emergencia. Toda la población probablemente se verá afectada.",
            "cautionary_statement" : "Todos deben evitar el tiempo al aire libre.",
            "color_code" : "#880E4F"
        }
    return statements


def generate_risk_prediction(aqi: int, data: Dict[str, Any]) -> str:
    """Genera una predicción de riesgo basada en el AQI y otros datos."""
    if aqi <= 100:
        return "La calidad del aire se mantendrá buena/moderada durante las próximas horas"
    elif aqi <= 150:
        return "Se espera que la calidad del aire empeore ligeramente en las próximas horas"
    else:
        return "Se recomienda precaución debido a la mala calidad del aire"

def get_pollutant_level(pollutant: str, value: float) -> dict:
    """
    Obtiene el nivel de calidad del aire para un contaminante específico.
    
    Args:
        pollutant (str): Código del contaminante (pm25, pm10, o3, no2, so2, co)
        value (float): Valor numérico del contaminante
    
    Returns:
        dict: Objeto con descripción y color hexadecimal
    """
    if pollutant == "pm25":
        if value <= 12:
            return {"description": "Bueno", "color": "#4CAF50"}
        elif value <= 35.4:
            return {"description": "Moderado", "color": "#FFEB3B"}
        elif value <= 55.4:
            return {"description": "Poco saludable para grupos sensibles", "color": "#FF9800"}
        elif value <= 150.4:
            return {"description": "Insalubre", "color": "#F44336"}
        elif value <= 250.4:
            return {"description": "Muy insalubre", "color": "#9C27B0"}
        else:
            return {"description": "Peligroso", "color": "#880E4F"}
    
    elif pollutant == "pm10":
        if value <= 54:
            return {"description": "Bueno", "color": "#4CAF50"}
        elif value <= 154:
            return {"description": "Moderado", "color": "#FFEB3B"}
        elif value <= 254:
            return {"description": "Poco saludable para grupos sensibles", "color": "#FF9800"}
        elif value <= 354:
            return {"description": "Insalubre", "color": "#F44336"}
        elif value <= 424:
            return {"description": "Muy insalubre", "color": "#9C27B0"}
        else:
            return {"description": "Peligroso", "color": "#880E4F"}
    
    elif pollutant == "o3":
        if value <= 54:
            return {"description": "Bueno", "color": "#4CAF50"}
        elif value <= 70:
            return {"description": "Moderado", "color": "#FFEB3B"}
        elif value <= 85:
            return {"description": "Poco saludable para grupos sensibles", "color": "#FF9800"}
        elif value <= 105:
            return {"description": "Insalubre", "color": "#F44336"}
        elif value <= 200:
            return {"description": "Muy insalubre", "color": "#9C27B0"}
        else:
            return {"description": "Peligroso", "color": "#880E4F"}
    
    elif pollutant == "no2":
        if value <= 53:
            return {"description": "Bueno", "color": "#4CAF50"}
        elif value <= 100:
            return {"description": "Moderado", "color": "#FFEB3B"}
        elif value <= 360:
            return {"description": "Poco saludable para grupos sensibles", "color": "#FF9800"}
        elif value <= 649:
            return {"description": "Insalubre", "color": "#F44336"}
        elif value <= 1249:
            return {"description": "Muy insalubre", "color": "#9C27B0"}
        else:
            return {"description": "Peligroso", "color": "#880E4F"}
    
    elif pollutant == "so2":
        if value <= 35:
            return {"description": "Bueno", "color": "#4CAF50"}
        elif value <= 75:
            return {"description": "Moderado", "color": "#FFEB3B"}
        elif value <= 185:
            return {"description": "Poco saludable para grupos sensibles", "color": "#FF9800"}
        elif value <= 304:
            return {"description": "Insalubre", "color": "#F44336"}
        elif value <= 604:
            return {"description": "Muy insalubre", "color": "#9C27B0"}
        else:
            return {"description": "Peligroso", "color": "#880E4F"}
    
    elif pollutant == "co":
        # CO generalmente se mide en mg/m³
        if value <= 4.4:
            return {"description": "Bueno", "color": "#4CAF50"}
        elif value <= 9.4:
            return {"description": "Moderado", "color": "#FFEB3B"}
        elif value <= 12.4:
            return {"description": "Poco saludable para grupos sensibles", "color": "#FF9800"}
        elif value <= 15.4:
            return {"description": "Insalubre", "color": "#F44336"}
        elif value <= 30.4:
            return {"description": "Muy insalubre", "color": "#9C27B0"}
        else:
            return {"description": "Peligroso", "color": "#880E4F"}
    
    else:
        return {"description": "Datos no disponibles", "color": "#9E9E9E"}