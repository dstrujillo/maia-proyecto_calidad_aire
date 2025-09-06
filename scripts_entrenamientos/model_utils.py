import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import mlflow
import mlflow.sklearn
import json
import joblib
import os

def load_dataset(feature_set):
    """Cargar dataset preprocesado"""
    file_path = f'../data/preprocessed_air_quality_data_feat_set{feature_set}.csv'
    df = pd.read_csv(file_path)
    
    # Separar características y variable objetivo
    X = df.drop('hospital_admissions', axis=1)
    y = df['hospital_admissions']
    
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Dividir datos en train, validation y test"""
    # Primera división: train + temp vs test
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Segunda división: train vs validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=val_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    """Escalar datos numéricos"""
    scaler = StandardScaler()
    
    # Ajustar solo con training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def get_model_config(model_name):
    """Configuración de modelos e hiperparámetros"""
    configs = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'selection': ['cyclic', 'random']
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'xgboost': {
            'model': XGBRegressor(random_state=42, verbosity=0),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'lightgbm': {
            'model': LGBMRegressor(random_state=42, verbosity=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [31, 50, 100]
            }
        },
        'svr': {
            'model': SVR(),
            'params': {
                'kernel': ['linear', 'rbf', 'poly'],
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 0.2]
            }
        },
        'mlp': {
            'model': MLPRegressor(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    }
    
    return configs.get(model_name, {})

def calculate_metrics(y_true, y_pred):
    """Calcular múltiples métricas de evaluación"""
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'median_absolute_error': np.median(np.abs(y_true - y_pred))
    }
    return metrics

def log_metrics(metrics, prefix=''):
    """Registrar métricas en MLflow"""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)

def train_model_with_mlflow(X_train, y_train, X_val, y_val, model_config, config_model_name,
                           feature_set, search_method='grid', cv=3):
    """Entrenar modelo con MLflow tracking"""
    
    model_name = model_config['model'].__class__.__name__
    run_name = f"{model_name}_set_{feature_set}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("search_method", search_method)
        
        # Búsqueda de hiperparámetros
        if search_method == 'grid':
            search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
        else:
            search = RandomizedSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0,
                n_iter=10,
                random_state=42
            )
        
        # Entrenamiento
        search.fit(X_train, y_train)
        
        # Mejor modelo
        best_model = search.best_estimator_
        
        # Log best parameters
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", -search.best_score_)
        
        # Predecir y calcular métricas
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        
        # Log metrics
        log_metrics(train_metrics, 'train')
        log_metrics(val_metrics, 'val')
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log artifacts
        artifacts_dir = f"../artifacts/set_{feature_set}/{config_model_name}"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Guardar modelo
        joblib.dump(best_model, f"{artifacts_dir}/model.joblib")
        
        # Guardar métricas
        with open(f"{artifacts_dir}/metrics.json", 'w') as f:
            json.dump({
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_params': search.best_params_
            }, f, indent=2)
        
        mlflow.log_artifact(f"{artifacts_dir}/metrics.json")
        
        return best_model, train_metrics, val_metrics, search.best_params_