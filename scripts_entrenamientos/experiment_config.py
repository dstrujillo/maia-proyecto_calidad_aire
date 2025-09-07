EXPERIMENT_CONFIG = {
    'feature_sets': [3],
    'models_to_train': [
        'linear_regression',
        'ridge',
        'lasso',
        #'random_forest',
        'gradient_boosting',
        'xgboost',
        'lightgbm',
        #'svr',
        #'mlp'
    ],
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42,
    'search_method': 'grid',  # 'grid' or 'random'
    'cv_folds': 3,
    'mlflow_tracking_uri': '../mlruns',
    'experiment_name': 'hospital_admissions_prediction_set3'
}

MODEL_PRIORITY = {
    'lightgbm': 1,
    'xgboost': 2,
    'random_forest': 3,
    'gradient_boosting': 4,
    'ridge': 5,
    'lasso': 6,
    'linear_regression': 7,
    'svr': 8,
    'mlp': 9
}