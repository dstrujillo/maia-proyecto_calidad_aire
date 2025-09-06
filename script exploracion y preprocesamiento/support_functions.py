import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

numeric_cols = ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'temperature', 
                    'humidity', 'hospital_admissions', 'hospital_capacity']

def detect_outliers(df, columns):
    """Detect outliers in specified columns of a DataFrame using the IQR method."""
    outlier_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100
        }
    return outlier_info


def show_outliers(data_df, numeric_columns):
    outlier_info = detect_outliers(data_df, numeric_columns)
    for col, info in outlier_info.items():
        print(f"{col}: {info['count']} valores atípicos ({info['percentage']:.2f}%)")

def impute_missing_values(df, numeric_cols):
    # Para variables numéricas, usar mediana por ciudad
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby('city')[col].transform(lambda x: x.fillna(x.median()))
    
    # Para categóricas, usar la moda por ciudad
    categorical_cols = ['population_density']
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby('city')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))
    
    return df

# Preparar diferentes opciones de escalado
def apply_scaling(df, scaling_method='standard'):
    
    # Crear copia para no modificar el original
    df_scaled = df.copy()
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Método de escalado no válido")
    
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler

def encode_categorical(df):
    # One-Hot Encoding para population_density
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_density = encoder.fit_transform(df[['population_density']])
    density_categories = encoder.categories_[0][1:]  # Excluir la primera categoría
    
    # Crear DataFrame con las nuevas columnas
    encoded_df = pd.DataFrame(encoded_density, 
                             columns=[f"density_{cat}" for cat in density_categories])
    
    # Codificar city (usaremos frecuencia en lugar de one-hot para muchas categorías)
    city_freq = df['city'].value_counts(normalize=True)
    df['city_freq'] = df['city'].map(city_freq)
    
    # Combinar con el DataFrame original
    df_encoded = pd.concat([df.drop(['population_density', 'city'], axis=1), encoded_df], axis=1)
    
    return df_encoded, encoder


# 4. Índice de calidad del aire categorizado
def categorize_aqi(aqi):
    if aqi <= 50:
        return 0  # Bueno
    elif aqi <= 100:
        return 1  # Moderado
    elif aqi <= 150:
        return 2  # Insalubre para grupos sensibles
    elif aqi <= 200:
        return 3  # Insalubre
    elif aqi <= 300:
        return 4  # Muy insalubre
    else:
        return 5  # Peligroso


def create_new_features(df):
    # Crear copia para no modificar el original
    df_new = df.copy()
    
    # 1. Hospital Utilization Ratio
    df_new['hospital_utilization'] = df_new['hospital_admissions'] / df_new['hospital_capacity']
    
    # 2. Variables temporales
    df_new['month'] = df_new['date'].dt.month
    df_new['day_of_week'] = df_new['date'].dt.dayofweek  # Lunes=0, Domingo=6
    df_new['is_weekend'] = df_new['day_of_week'].isin([5, 6]).astype(int)
    df_new['season'] = df_new['month'].apply(lambda x: (x % 12 + 3) // 3)  # 1:Invierno, 2:Primavera, etc.
    
    # 3. Interacciones entre contaminantes
    df_new['pm_interaction'] = df_new['pm2_5'] * df_new['pm10']
    df_new['no2_o3_ratio'] = df_new['no2'] / (df_new['o3'] + 1)  # +1 para evitar división por cero
    

    df_new['aqi_category'] = df_new['aqi'].apply(categorize_aqi)
    
    # 5. Estrés térmico (combinación de temperatura y humedad)
    df_new['thermal_stress'] = df_new['temperature'] * (df_new['humidity'] / 100)
    
    return df_new


def create_feature_set_1(df):
    """Opción 1: Características básicas + temporales"""
    features = [
        'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 
        'temperature', 'humidity', 
        'month', 'day_of_week', 'is_weekend', 'season',
        'hospital_utilization'
    ]
    
    # Añadir características de densidad poblacional codificadas
    density_features = [col for col in df.columns if col.startswith('density_')]
    features.extend(density_features)
    
    # Añadir frecuencia de ciudad
    features.append('city_freq')
    
    return df[features]

def create_feature_set_2(df):
    """Opción 2: Características básicas + interacciones"""
    features = [
        'aqi', 'pm2_5', 'pm10', 'no2', 'o3', 
        'temperature', 'humidity', 
        'pm_interaction', 'no2_o3_ratio', 'thermal_stress',
        'hospital_utilization'
    ]
    
    # Añadir características de densidad poblacional codificadas
    density_features = [col for col in df.columns if col.startswith('density_')]
    features.extend(density_features)
    
    # Añadir frecuencia de ciudad
    features.append('city_freq')
    
    return df[features]


def create_feature_set_3(df):
    """Opción 3: Todas las características"""
    # Excluir columnas no relevantes
    exclude_cols = ['date', 'hospital_admissions']
    features = [col for col in df.columns if col not in exclude_cols]
    
    return df[features]