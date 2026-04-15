import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

# 1. CARREGAMENTO DOS DADOS E CONFIGURAÇÃO DE DIRETÓRIOS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'datasets'
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'dataset_final.csv')

# Lista de transtornos para processar (removendo a depressão que já foi feita)
targets = {
    'Preval_Ansiedade': 'ansiedade',
    'Preval_T_Alimentar': 't_alimentar',
    'Preval_Bipolaridade': 'bipolaridade',
    'Preval_Esquizofrenia': 'esquizofrenia'
}

# 2. DEFINIÇÃO DAS FEATURES (X) 
# Removemos IDs, Ano e todas as colunas de prevalência (alvos) do X
all_target_cols = ['Preval_Esquizofrenia', 'Preval_Depressao', 'Preval_Ansiedade', 
                   'Preval_Bipolaridade', 'Preval_T_Alimentar']

X = df.drop(columns=['Entity', 'Code', 'Year'] + all_target_cols)

# 3. LOOP DE PROCESSAMENTO PARA CADA TRANSTORNO
for target_col, name in targets.items():
    print(f"Processando: {target_col}...")
    y = df[target_col]

    # 3.1 CONFIGURAÇÃO DO MODELO E K-FOLD
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Validação Cruzada
    scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
    results = cross_validate(model, X, y, cv=kf, scoring=scoring)

    # 4. GRÁFICO DE MÉTRICAS (DESEMPENHO)
    metrics_df = pd.DataFrame({
        'Métrica': ['R²', 'MAE', 'RMSE'],
        'Valor': [results['test_r2'].mean(), -results['test_mae'].mean(), -results['test_rmse'].mean()]
    })

    plt.figure(figsize=(8, 4))
    sns.barplot(x='Valor', y='Métrica', data=metrics_df, palette='viridis', hue='Métrica', legend=False)
    plt.title(f'Desempenho Global do Modelo - {target_col}')
    plt.xlim(0, max(1.1, metrics_df['Valor'].max() * 1.1))
    for i, v in enumerate(metrics_df['Valor']):
        plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'metricas_desempenho_{name}.png')
    plt.close() # Fecha a figura para não acumular memória

    # 5. GRÁFICO DE IMPORTÂNCIA (RANKING)
    model.fit(X, y)
    importances = pd.DataFrame({
        'Atributo': X.columns,
        'Importancia': model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importancia', y='Atributo', data=importances, palette='magma', hue='Atributo', legend=False)
    plt.title(f'Ranking de Importância de Atributos - {target_col}', fontsize=14)
    plt.xlabel('Importância Relativa (MDI)', fontsize=12)
    plt.ylabel('Atributos', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'ranking_importancia_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\nProcessamento concluído com sucesso!")
print(f"Os gráficos foram salvos em: {PLOTS_DIR}")