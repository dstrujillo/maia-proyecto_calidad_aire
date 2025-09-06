from support_functions import *

def full_preprocessing_pipeline(df, scaling_method='standard', feature_set=1):
    """
    Pipeline completo de preprocesamiento
    
    Parámetros:
    - df: DataFrame original
    - scaling_method: 'standard', 'minmax' o 'robust'
    - feature_set: 1, 2 o 3 (selección del conjunto de características)
    """
    
    # Convertir date a datetime
    df['date'] = pd.to_datetime(df['date'])

    # Convertir city a categórico
    df['city'] = df['city'].astype('category')
    # Paso 1: Crear nuevas características
    df_processed = create_new_features(df)
    
    # Paso 2: Codificar variables categóricas
    df_processed, encoder = encode_categorical(df_processed)
    
    # Paso 3: Escalar variables numéricas
    df_scaled, scaler = apply_scaling(df_processed, scaling_method)
    
    # Paso 4: Seleccionar conjunto de características
    if feature_set == 1:
        X = create_feature_set_1(df_scaled)
    elif feature_set == 2:
        X = create_feature_set_2(df_scaled)
    elif feature_set == 3:
        X = create_feature_set_3(df_scaled)
    else:
        raise ValueError("El feature_set debe ser 1, 2 o 3")
    
    # Variable objetivo
    y = df_scaled['hospital_admissions']
    
    return X, y, encoder, scaler