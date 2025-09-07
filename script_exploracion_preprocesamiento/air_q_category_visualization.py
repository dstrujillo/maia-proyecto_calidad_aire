import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr
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
    
    # Verificar si existe la columna air_quality_category
    if 'air_quality_category' not in df.columns:
        # Crear la categoría de calidad del aire basada en AQI si no existe
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
        
        if 'aqi' in df.columns:
            df['air_quality_category'] = df['aqi'].apply(categorize_aqi)
        else:
            raise ValueError("No se encuentra la columna 'aqi' para crear la categoría de calidad del aire")
    
    return df

def calculate_correlation_matrix(df, target_var='air_quality_category', method='spearman'):
    """
    Calcular matriz de correlación con la variable objetivo categórica
    """
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_var not in numeric_cols:
        raise ValueError(f"La variable objetivo '{target_var}' no está en el DataFrame o no es numérica")
    
    # Para variables categóricas ordinales, usar Spearman
    if method == 'spearman':
        corr_matrix = df[numeric_cols].corr(method='spearman')
    elif method == 'pointbiserial':
        # Para correlación punto-biserial (variable categórica vs continuas)
        corr_matrix = df[numeric_cols].corr(method='spearman')  # Usamos Spearman como aproximación
    else:
        corr_matrix = df[numeric_cols].corr()
    
    return corr_matrix

def plot_correlation_matrix(corr_matrix, target_var, feature_set, method='spearman'):
    """
    Visualizar matriz de correlación con la variable objetivo categórica
    """
    # Ordenar correlaciones con la variable objetivo
    target_correlations = corr_matrix[target_var].sort_values(ascending=False)
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Gráfico 1: Correlaciones con la variable objetivo (barras)
    colors = ['red' if x < 0 else 'blue' for x in target_correlations]
    bars = ax1.barh(range(len(target_correlations)), target_correlations.values, color=colors)
    ax1.set_yticks(range(len(target_correlations)))
    ax1.set_yticklabels(target_correlations.index, fontsize=9)
    ax1.set_xlabel('Coeficiente de Correlación')
    ax1.set_title(f'Correlación con {target_var} - Set {feature_set} ({method})', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(axis='x', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, target_correlations.values)):
        ax1.text(value + (0.01 if value >= 0 else -0.03), i, f'{value:.3f}', 
                va='center', ha='left' if value >= 0 else 'right', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Gráfico 2: Heatmap de correlaciones (focused on target variable)
    # Filtrar solo las variables con mayor correlación absoluta
    top_vars = target_correlations[abs(target_correlations) > 0.1].index.tolist()
    if len(top_vars) > 1:
        corr_subset = corr_matrix.loc[top_vars, top_vars]
    else:
        corr_subset = corr_matrix
    
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    heatmap = sns.heatmap(corr_subset, 
                         annot=True, 
                         fmt='.2f', 
                         cmap='RdBu_r', 
                         center=0,
                         square=True, 
                         mask=mask,
                         ax=ax2,
                         cbar_kws={'shrink': 0.8})
    ax2.set_title(f'Matriz de Correlación (Top Variables) - Set {feature_set}', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'../visualizations/correlation_matrix_aqi_category_set_{feature_set}_{method}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return target_correlations

def analyze_aqi_correlations(target_correlations, threshold=0.3):
    """
    Analizar correlaciones significativas con la categoría de calidad del aire
    """
    print("\n" + "="*70)
    print("ANÁLISIS DE CORRELACIONES CON CATEGORÍA DE CALIDAD DEL AIRE")
    print("="*70)
    
    # Excluir la auto-correlación
    target_correlations = target_correlations[target_correlations.index != 'air_quality_category']
    
    # Correlaciones positivas fuertes (variables que aumentan con peor calidad del aire)
    strong_positive = target_correlations[target_correlations > threshold]
    
    # Correlaciones negativas fuertes (variables que disminuyen con peor calidad del aire)
    strong_negative = target_correlations[target_correlations < -threshold]
    
    print(f"\nVariables que AUMENTAN con peor calidad del aire (> {threshold}):")
    if len(strong_positive) > 0:
        for var, corr in strong_positive.items():
            print(f"  {var}: {corr:.3f}")
    else:
        print("  No se encontraron correlaciones positivas fuertes")
    
    print(f"\nVariables que DISMINUYEN con peor calidad del aire (< -{threshold}):")
    if len(strong_negative) > 0:
        for var, corr in strong_negative.items():
            print(f"  {var}: {corr:.3f}")
    else:
        print("  No se encontraron correlaciones negativas fuertes")
    
    return strong_positive, strong_negative

def generate_aqi_category_report(corr_matrix, target_var, feature_set):
    """
    Generar reporte detallado de correlaciones con la categoría de calidad del aire
    """
    target_correlations = corr_matrix[target_var].sort_values(ascending=False)
    
    # Excluir la auto-correlación
    target_correlations = target_correlations[target_correlations.index != target_var]
    
    print(f"\n{'='*90}")
    print(f"REPORTE DE CORRELACIÓN CON CATEGORÍA DE CALIDAD DEL AIRE - CONJUNTO {feature_set}")
    print(f"{'='*90}")
    
    print(f"\nTop 10 variables PREDICTORAS de peor calidad del aire:")
    top_positive = target_correlations.head(10)
    for i, (var, corr) in enumerate(top_positive.items(), 1):
        print(f"  {i:2d}. {var:25s}: {corr:.3f}")
    
    print(f"\nTop 10 variables asociadas con MEJOR calidad del aire:")
    top_negative = target_correlations.tail(10)
    for i, (var, corr) in enumerate(top_negative[::-1].items(), 1):
        print(f"  {i:2d}. {var:25s}: {corr:.3f}")
    
    # Estadísticas generales
    print(f"\nESTADÍSTICAS DE CORRELACIÓN:")
    print(f"  Correlación máxima: {target_correlations.max():.3f}")
    print(f"  Correlación mínima: {target_correlations.min():.3f}")
    print(f"  Correlación promedio: {target_correlations.mean():.3f}")
    print(f"  Número de variables con |correlación| > 0.3: {len(target_correlations[abs(target_correlations) > 0.3])}")
    print(f"  Número de variables con |correlación| > 0.5: {len(target_correlations[abs(target_correlations) > 0.5])}")

def plot_aqi_category_distribution(df, feature_set):
    """
    Visualizar la distribución de la categoría de calidad del aire
    """
    if 'air_quality_category' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Mapear categorías a nombres descriptivos
        category_names = {
            0: 'Buena (0-50)',
            1: 'Moderada (51-100)',
            2: 'Insalubre Grupos Sensibles (101-150)',
            3: 'Insalubre (151-200)',
            4: 'Muy Insalubre (201-300)',
            5: 'Peligrosa (301-500)'
        }
        
        # Contar frecuencias
        counts = df['air_quality_category'].value_counts().sort_index()
        labels = [category_names.get(i, f'Categoría {i}') for i in counts.index]
        
        # Crear gráfico de barras
        bars = plt.bar(range(len(counts)), counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(counts))))
        plt.xlabel('Categoría de Calidad del Aire')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de Categorías de Calidad del Aire - Set {feature_set}', fontweight='bold')
        plt.xticks(range(len(counts)), labels, rotation=45, ha='right')
        
        # Añadir valores en las barras
        for i, (bar, count) in enumerate(zip(bars, counts.values)):
            plt.text(i, count + max(counts.values)*0.01, f'{count}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'../visualizations/aqi_category_distribution_set_{feature_set}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nDistribución de categorías de calidad del aire:")
        for i, count in counts.items():
            print(f"  {category_names.get(i, f'Categoría {i}')}: {count} muestras ({count/len(df)*100:.1f}%)")

def compare_aqi_correlations_across_sets():
    """
    Comparar correlaciones con AQI category entre los tres conjuntos
    """
    comparison_data = {}
    
    for feature_set in range(1, 4):
        df = load_and_prepare_data(feature_set)
        corr_matrix = calculate_correlation_matrix(df, 'air_quality_category')
        target_correlations = corr_matrix['air_quality_category'].sort_values(ascending=False)
        comparison_data[f'Set {feature_set}'] = target_correlations
    
    # Crear DataFrame comparativo
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df[comparison_df.index != 'air_quality_category']
    
    # Graficar comparación
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(comparison_df.columns):
        sorted_corr = comparison_df[col].sort_values(ascending=False)
        ax = axes[i]
        
        # Seleccionar top 15 variables
        top_vars = sorted_corr.head(15)
        colors = ['red' if x < 0 else 'blue' for x in top_vars.values]
        
        bars = ax.barh(range(len(top_vars)), top_vars.values, color=colors)
        ax.set_yticks(range(len(top_vars)))
        ax.set_yticklabels(top_vars.index, fontsize=8)
        ax.set_xlabel('Coeficiente de Correlación')
        ax.set_title(f'Top 15 Correlaciones - {col}', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='x', alpha=0.3)
        
        # Añadir valores en las barras
        for j, (bar, value) in enumerate(zip(bars, top_vars.values)):
            ax.text(value + (0.01 if value >= 0 else -0.03), j, f'{value:.3f}', 
                   va='center', ha='left' if value >= 0 else 'right', fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # Gráfico de comparación de las top 5 variables más correlacionadas
    top_vars_comparison = {}
    for col in comparison_df.columns:
        top_vars_comparison[col] = comparison_df[col].head(5).index.tolist()
    
    # Encontrar variables comunes
    all_vars = set()
    for vars_list in top_vars_comparison.values():
        all_vars.update(vars_list)
    
    common_vars = list(all_vars)
    comparison_common = comparison_df.loc[common_vars]
    
    axes[3].axis('off')  # Ocultar el cuarto subplot
    
    plt.tight_layout()
    plt.savefig('../visualizations/aqi_category_correlation_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

# Ejecutar análisis para los tres conjuntos de características
all_aqi_correlations = {}

print("ANÁLISIS DE CORRELACIÓN CON CATEGORÍA DE CALIDAD DEL AIRE")
print("="*80)

for feature_set in range(1, 4):
    print(f"\n{'#'*80}")
    print(f"ANALIZANDO CONJUNTO DE CARACTERÍSTICAS {feature_set}")
    print(f"{'#'*80}")
    
    try:
        # Cargar datos
        df = load_and_prepare_data(feature_set)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Mostrar distribución de la categoría de calidad del aire
        plot_aqi_category_distribution(df, feature_set)
        
        # Calcular matriz de correlación (Spearman para variables ordinales)
        corr_matrix = calculate_correlation_matrix(df, 'air_quality_category', 'spearman')
        
        # Visualizar y analizar
        target_correlations = plot_correlation_matrix(corr_matrix, 'air_quality_category', feature_set, 'spearman')
        strong_positive, strong_negative = analyze_aqi_correlations(target_correlations, threshold=0.3)
        generate_aqi_category_report(corr_matrix, 'air_quality_category', feature_set)
        
        # Guardar para comparación posterior
        all_aqi_correlations[f'Set {feature_set}'] = target_correlations
        
        # Guardar matrices de correlación
        corr_matrix.to_csv(f'../results/aqi_category_correlation_matrix_set_{feature_set}_spearman.csv')
        
    except Exception as e:
        print(f"Error procesando el conjunto {feature_set}: {e}")
        continue

# Comparar los tres conjuntos
print(f"\n{'#'*80}")
print("COMPARACIÓN ENTRE LOS TRES CONJUNTOS - CATEGORÍA AQI")
print(f"{'#'*80}")

comparison_df = compare_aqi_correlations_across_sets()

# Identificar las variables más consistentemente correlacionadas
consistent_correlations = {}
for variable in comparison_df.index:
    correlations = comparison_df.loc[variable]
    if all(abs(corr) > 0.2 for corr in correlations):  # Correlación moderada en los tres sets
        consistent_correlations[variable] = {
            'avg_correlation': correlations.mean(),
            'std_correlation': correlations.std(),
            'min_correlation': correlations.min(),
            'max_correlation': correlations.max()
        }

print(f"\nVariables consistentemente correlacionadas con la categoría AQI:")
if consistent_correlations:
    consistent_df = pd.DataFrame(consistent_correlations).T.sort_values('avg_correlation', ascending=False)
    for var, stats in consistent_df.head(10).iterrows():
        print(f"  {var:25s}: {stats['avg_correlation']:.3f} (avg) ± {stats['std_correlation']:.3f}")
else:
    print("  No se encontraron variables con correlación consistente en los tres conjuntos")

# Guardar resultados de la comparación
comparison_df.to_csv('../results/aqi_category_correlation_comparison.csv')
consistent_df.to_csv('../results/aqi_consistent_correlations.csv')

print("\nAnálisis completado. Todos los gráficos y resultados han sido guardados.")
print("Archivos guardados en las carpetas '../visualizations/' y '../results/'")