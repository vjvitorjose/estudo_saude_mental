import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

# 1. CONFIGURAÇÃO DE CAMINHOS (Baseado no seu código)
DATA_DIR = Path(__file__).resolve().parent.parent / 'datasets'
PLOTS_DIR = Path(__file__).resolve().parent.parent / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Carregamento
df = pd.read_csv(DATA_DIR / 'dataset_final.csv')
target = 'Preval_Ansiedade'

# 2. DEFINIÇÃO DE X E Y
# Removemos identificadores e as outras doenças para evitar "vazamento"
X = df.drop(columns=['Entity', 'Code', 'Year', 
                     'Preval_Esquizofrenia', 'Preval_Depressao', 
                     'Preval_Ansiedade', 'Preval_Bipolaridade', 
                     'Preval_T_Alimentar'])
y = df[target]

# 3. CRIAÇÃO DO CUSTOM K-FOLD (Equilibrado por País)
# Queremos que cada fold contenha a mesma representatividade de cada país.
# Vamos criar 5 pares de (índices_treino, índices_teste) manualmente.
n_splits = 5
custom_cv = [([], []) for _ in range(n_splits)]

# Iteramos por cada país para distribuir suas linhas igualmente entre os folds
for code, group in df.groupby('Code'):
    indices = group.index.values
    # Criamos um split interno para o país
    kf_country = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for i, (train_sub_idx, test_sub_idx) in enumerate(kf_country.split(indices)):
        # Adicionamos os índices reais do dataset ao fold correspondente
        custom_cv[i][0].extend(indices[train_sub_idx])
        custom_cv[i][1].extend(indices[test_sub_idx])

# 4. CONFIGURAÇÃO E VALIDAÇÃO CRUZADA
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}

# Rodamos a validação usando nosso custom_cv
results = cross_validate(model, X, y, cv=custom_cv, scoring=scoring)

# 5. GRÁFICO DE MÉTRICAS (DESEMPENHO)
metrics_df = pd.DataFrame({
    'Métrica': ['R²', 'MAE', 'RMSE'],
    'Valor': [results['test_r2'].mean(), -results['test_mae'].mean(), -results['test_rmse'].mean()]
})

plt.figure(figsize=(8, 4))
sns.barplot(x='Valor', y='Métrica', data=metrics_df, palette='viridis', hue='Métrica', legend=False)
plt.title(f'Desempenho Global (Fold Equilibrado) - {target}')
plt.xlim(0, max(1.1, metrics_df['Valor'].max() * 1.1))
for i, v in enumerate(metrics_df['Valor']):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'metricas_desempenho_ansiedade.png')

# 6. GRÁFICO DE IMPORTÂNCIA (RANKING)
model.fit(X, y)
importances = pd.DataFrame({
    'Atributo': X.columns,
    'Importancia': model.feature_importances_
}).sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importancia', y='Atributo', data=importances, palette='magma', hue='Atributo', legend=False)
plt.title(f'Ranking de Importância de Atributos - {target}', fontsize=14)
plt.xlabel('Importância Relativa (MDI)', fontsize=12)
plt.ylabel('Atributos', fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ranking_importancia_ansiedade.png', dpi=150, bbox_inches='tight')

print(f"Processamento concluído para {target}.")