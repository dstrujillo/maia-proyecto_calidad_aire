import argparse
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import sys
import joblib

# Añadir directorio actual al path para importar módulos
sys.path.append('.')

from model_utils import (
    load_dataset, split_data, scale_data, 
    get_model_config, train_model_with_mlflow, calculate_metrics
)
from experiment_config import EXPERIMENT_CONFIG, MODEL_PRIORITY

def setup_mlflow():
    """Configurar MLflow"""
    mlflow.set_tracking_uri(EXPERIMENT_CONFIG['mlflow_tracking_uri'])
    mlflow.set_experiment(EXPERIMENT_CONFIG['experiment_name'])
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_CONFIG['experiment_name'])
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_CONFIG['experiment_name'])
    else:
        experiment_id = experiment.experiment_id
    
    return client, experiment_id

def run_experiment():
    """Ejecutar experimento completo"""
    
    # Configurar MLflow
    client, experiment_id = setup_mlflow()
    
    # Resultados globales
    all_results = []
    
    for feature_set in EXPERIMENT_CONFIG['feature_sets']:
        print(f"\n{'='*80}")
        print(f"ENTRENANDO CON FEATURE SET {feature_set}")
        print(f"{'='*80}")
        
        try:
            # Cargar datos
            X, y = load_dataset(feature_set)
            print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
            
            # Dividir datos
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                X, y, 
                test_size=EXPERIMENT_CONFIG['test_size'],
                val_size=EXPERIMENT_CONFIG['val_size'],
                random_state=EXPERIMENT_CONFIG['random_state']
            )
            
            print(f"Divisiones: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
            
            # Escalar datos
            X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(X_train, X_val, X_test)
            
            # Guardar datos de test para evaluación final
            test_data_dir = f"../artifacts/set_{feature_set}/test_data"
            os.makedirs(test_data_dir, exist_ok=True)
            np.save(f"{test_data_dir}/X_test.npy", X_test_scaled)
            np.save(f"{test_data_dir}/y_test.npy", y_test.values)
            joblib.dump(scaler, f"{test_data_dir}/scaler.joblib")
            
            # Ordenar modelos por prioridad
            models_sorted = sorted(
                EXPERIMENT_CONFIG['models_to_train'],
                key=lambda x: MODEL_PRIORITY.get(x, 10)
            )
            
            for model_name in models_sorted:
                print(f"\n--- Entrenando {model_name} ---")
                
                try:
                    # Obtener configuración del modelo
                    model_config = get_model_config(model_name)
                    if not model_config:
                        print(f"Configuración no encontrada para {model_name}")
                        continue
                    
                    # Entrenar modelo
                    start_time = time.time()
                    
                    best_model, train_metrics, val_metrics, best_params = train_model_with_mlflow(
                        X_train_scaled, y_train, X_val_scaled, y_val,
                        model_config, model_name, feature_set,
                        search_method=EXPERIMENT_CONFIG['search_method'],
                        cv=EXPERIMENT_CONFIG['cv_folds']
                    )
                    
                    training_time = time.time() - start_time
                    
                    # Guardar resultados
                    result = {
                        'feature_set': feature_set,
                        'model': model_name,
                        'training_time': training_time,
                        'train_rmse': train_metrics['rmse'],
                        'val_rmse': val_metrics['rmse'],
                        'train_r2': train_metrics['r2'],
                        'val_r2': val_metrics['r2'],
                        'best_params': str(best_params)
                    }
                    
                    all_results.append(result)
                    
                    print(f"✅ {model_name} entrenado en {training_time:.2f}s")
                    print(f"   Train RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
                    print(f"   Val RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
                    
                except Exception as e:
                    print(f"❌ Error entrenando {model_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error procesando feature set {feature_set}: {e}")
            continue
    
    # Guardar resultados globales
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('../results/training_results.csv', index=False)
    
    # Mostrar mejores modelos por feature set
    print(f"\n{'='*80}")
    print("MEJORES MODELOS POR FEATURE SET")
    print(f"{'='*80}")
    
    for feature_set in EXPERIMENT_CONFIG['feature_sets']:
        set_results = results_df[results_df['feature_set'] == feature_set]
        if not set_results.empty:
            best_model = set_results.loc[set_results['val_rmse'].idxmin()]
            print(f"Set {feature_set}: {best_model['model']} - Val RMSE: {best_model['val_rmse']:.4f}")
    
    return results_df

def evaluate_best_models():
    """Evaluar los mejores modelos en el conjunto de test"""
    
    print(f"\n{'='*80}")
    print("EVALUACIÓN FINAL EN TEST")
    print(f"{'='*80}")
    
    final_results = []
    
    for feature_set in EXPERIMENT_CONFIG['feature_sets']:
        # Cargar datos de test
        test_data_dir = f"../artifacts/set_{feature_set}/test_data"
        try:
            X_test = np.load(f"{test_data_dir}/X_test.npy")
            y_test = np.load(f"{test_data_dir}/y_test.npy")
            scaler = joblib.load(f"{test_data_dir}/scaler.joblib")
        except:
            print(f"❌ No se encontraron datos de test para set {feature_set}")
            continue
        
        # Buscar el mejor modelo para este feature set
        results_df = pd.read_csv('../results/training_results.csv')
        set_results = results_df[results_df['feature_set'] == feature_set]
        
        if set_results.empty:
            print(f"❌ No hay resultados para set {feature_set}")
            continue
        
        best_model_info = set_results.loc[set_results['val_rmse'].idxmin()]
        model_name = best_model_info['model']
        
        # Cargar el modelo entrenado
        model_path = f"../artifacts/set_{feature_set}/{model_name}/model.joblib"
        try:
            model = joblib.load(model_path)
            
            # Predecir en test
            y_test_pred = model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_test_pred)
            
            # Registrar en MLflow
            with mlflow.start_run(run_name=f"test_{model_name}_set_{feature_set}"):
                mlflow.log_param("feature_set", feature_set)
                mlflow.log_param("model", model_name)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                mlflow.sklearn.log_model(model, "best_model")
            
            # Guardar resultados
            result = {
                'feature_set': feature_set,
                'model': model_name,
                'test_rmse': test_metrics['rmse'],
                'test_r2': test_metrics['r2'],
                'test_mae': test_metrics['mae'],
                'test_mape': test_metrics['mape']
            }
            
            final_results.append(result)
            
            print(f"Set {feature_set} - {model_name}:")
            print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"  Test R²: {test_metrics['r2']:.4f}")
            print(f"  Test MAE: {test_metrics['mae']:.4f}")
            print(f"  Test MAPE: {test_metrics['mape']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluando {model_name} en set {feature_set}: {e}")
            continue
    
    # Guardar resultados finales
    final_df = pd.DataFrame(final_results)
    final_df.to_csv('../results/final_test_results.csv', index=False)
    
    return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelos para predecir hospital admissions')
    parser.add_argument('--evaluate-only', action='store_true', 
                       help='Solo evaluar modelos ya entrenados')
    
    args = parser.parse_args()
    
    if not args.evaluate_only:
        print("Iniciando entrenamiento de modelos...")
        results = run_experiment()
    
    print("\nEvaluando mejores modelos en conjunto de test...")
    final_results = evaluate_best_models()
    
    print("\n🎯 Entrenamiento y evaluación completados!")
    print("Resultados guardados en:")
    print("- ../results/training_results.csv")
    print("- ../results/final_test_results.csv")
    print("- MLflow: ../mlruns/")