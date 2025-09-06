import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las visualizaciones
plt.style.use('default')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_prepare_data(feature_set):
    """
    Cargar y preparar datos para el conjunto de características especificado
    """
    file_path = f'../data/preprocessed_air_quality_data_feat_set{feature_set}.csv'
    df = pd.read_csv(file_path)
    return df

def calculate_correlation_matrix(df, target_var='hospital_admissions', method='pearson'):
    """
    Calcular matriz de correlación con la variable objetivo
    """
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_var not in numeric_cols:
        raise ValueError(f"La variable objetivo '{target_var}' no está en el DataFrame o no es numérica")
    
    # Calcular matriz de correlación
    if method == 'spearman':
        corr_matrix = df[numeric_cols].corr(method='spearman')
    else:
        corr_matrix = df[numeric_cols].corr()
    
    return corr_matrix

def plot_correlation_matrix(corr_matrix, target_var, feature_set, method='pearson'):
    """
    Visualizar matriz de correlación con la variable objetivo
    """
    # Ordenar correlaciones con la variable objetivo
    target_correlations = corr_matrix[target_var].sort_values(ascending=False)
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico 1: Correlaciones con la variable objetivo (barras)
    colors = ['red' if x < 0 else 'blue' for x in target_correlations]
    bars = ax1.barh(range(len(target_correlations)), target_correlations.values, color=colors)
    ax1.set_yticks(range(len(target_correlations)))
    ax1.set_yticklabels(target_correlations.index, fontsize=9)
    ax1.set_xlabel('Coeficiente de Correlación')
    ax1.set_title(f'Correlación con {target_var} - Set {feature_set} ({method})')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, target_correlations.values)):
        ax1.text(value + (0.01 if value >= 0 else -0.03), i, f'{value:.3f}', 
                va='center', ha='left' if value >= 0 else 'right', fontsize=8)
    
    # Gráfico 2: Heatmap de correlaciones
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara para el triángulo superior
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r', 
                center=0,
                square=True, 
                mask=mask,
                ax=ax2,
                cbar_kws={'shrink': 0.8})
    ax2.set_title(f'Matriz de Correlación - Set {feature_set} ({method})')
    
    plt.tight_layout()
    plt.savefig(f'../visualizations/correlation_matrix_set_{feature_set}_{method}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return target_correlations

def analyze_correlations(target_correlations, threshold=0.3):
    """
    Analizar correlaciones significativas
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE CORRELACIONES SIGNIFICATIVAS")
    print("="*60)
    
    # Correlaciones positivas fuertes
    strong_positive = target_correlations[target_correlations > threshold]
    strong_positive = strong_positive[strong_positive.index != 'hospital_admissions']
    
    # Correlaciones negativas fuertes
    strong_negative = target_correlations[target_correlations < -threshold]
    
    print(f"\nVariables con correlación POSITIVA fuerte (> {threshold}):")
    if len(strong_positive) > 0:
        for var, corr in strong_positive.items():
            print(f"  {var}: {corr:.3f}")
    else:
        print("  No se encontraron correlaciones positivas fuertes")
    
    print(f"\nVariables con correlación NEGATIVA fuerte (< -{threshold}):")
    if len(strong_negative) > 0:
        for var, corr in strong_negative.items():
            print(f"  {var}: {corr:.3f}")
    else:
        print("  No se encontraron correlaciones negativas fuertes")
    
    return strong_positive, strong_negative

def generate_correlation_report(corr_matrix, target_var, feature_set):
    """
    Generar reporte detallado de correlaciones
    """
    target_correlations = corr_matrix[target_var].sort_values(ascending=False)
    
    # Excluir la auto-correlación
    target_correlations = target_correlations[target_correlations.index != target_var]
    
    print(f"\n{'='*80}")
    print(f"REPORTE DE CORRELACIÓN - CONJUNTO DE CARACTERÍSTICAS {feature_set}")
    print(f"{'='*80}")
    
    print(f"\nTop 5 correlaciones POSITIVAS con {target_var}:")
    top_positive = target_correlations.head(5)
    for var, corr in top_positive.items():
        print(f"  {var}: {corr:.3f}")
    
    print(f"\nTop 5 correlaciones NEGATIVAS con {target_var}:")
    top_negative = target_correlations.tail(5)
    for var, corr in top_negative[::-1].items():
        print(f"  {var}: {corr:.3f}")
    
    # Estadísticas generales
    print(f"\nEstadísticas de correlación:")
    print(f"  Correlación máxima: {target_correlations.max():.3f}")
    print(f"  Correlación mínima: {target_correlations.min():.3f}")
    print(f"  Correlación promedio: {target_correlations.mean():.3f}")
    print(f"  Número de variables con |correlación| > 0.3: {len(target_correlations[abs(target_correlations) > 0.3])}")
    print(f"  Número de variables con |correlación| > 0.5: {len(target_correlations[abs(target_correlations) > 0.5])}")

def compare_feature_sets_correlations():
    """
    Comparar correlaciones entre los tres conjuntos de características
    """
    comparison_data = {}
    
    for feature_set in range(1, 4):
        df = load_and_prepare_data(feature_set)
        corr_matrix = calculate_correlation_matrix(df, 'hospital_admissions')
        target_correlations = corr_matrix['hospital_admissions'].sort_values(ascending=False)
        comparison_data[f'Set {feature_set}'] = target_correlations
    
    # Crear DataFrame comparativo
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df[comparison_df.index != 'hospital_admissions']
    
    # Graficar comparación
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(comparison_df.columns):
        sorted_corr = comparison_df[col].sort_values(ascending=False)
        plt.subplot(2, 2, i+1)
        colors = ['red' if x < 0 else 'blue' for x in sorted_corr.values]
        bars = plt.barh(range(len(sorted_corr)), sorted_corr.values, color=colors)
        plt.yticks(range(len(sorted_corr)), sorted_corr.index, fontsize=8)
        plt.xlabel('Coeficiente de Correlación')
        plt.title(f'Correlaciones con hospital_admissions - {col}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Añadir valores en las barras para las top 10
        for j, (bar, value) in enumerate(zip(bars, sorted_corr.values)):
            if j < 10:  # Solo mostrar valores para las top 10
                plt.text(value + (0.01 if value >= 0 else -0.03), j, f'{value:.3f}', 
                        va='center', ha='left' if value >= 0 else 'right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('../visualizations/correlation_comparison_all_sets.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df
