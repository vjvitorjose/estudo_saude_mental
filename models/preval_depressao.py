import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

# 1. CARREGAMENTO DOS DADOS
DATA_DIR = Path(__file__).resolve().parent.parent / 'datasets'
PLOTS_DIR = Path(__file__).resolve().parent.parent / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA_DIR / 'dataset_final.csv')

target = 'Preval_Depressao'

# 2. DEFINIÇÃO DO ALVO E ATRIBUTOS
X = df.drop(columns=['Entity', 'Code', 'Year', 
                     'Preval_Esquizofrenia', 'Preval_Depressao', 
                     'Preval_Ansiedade', 'Preval_Bipolaridade', 
                     'Preval_T_Alimentar'])
y = df[target]

# 3. CONFIGURAÇÃO E VALIDAÇÃO CRUZADA
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
results = cross_validate(model, X, y, cv=kf, scoring=scoring)

# 4. GRÁFICO DE MÉTRICAS (DESEMPENHO)
metrics_df = pd.DataFrame({
    'Métrica': ['R²', 'MAE', 'RMSE'],
    'Valor': [results['test_r2'].mean(), -results['test_mae'].mean(), -results['test_rmse'].mean()]
})

plt.figure(figsize=(8, 4))
sns.barplot(x='Valor', y='Métrica', data=metrics_df, palette='viridis', hue='Métrica', legend=False)
plt.title(f'Desempenho Global do Modelo - {target}')
plt.xlim(0, max(1.1, metrics_df['Valor'].max() * 1.1)) # Espaço para o texto
for i, v in enumerate(metrics_df['Valor']):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'metricas_desempenho_depressao.png')

# 5. GRÁFICO DE IMPORTÂNCIA (RANKING)
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
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ranking_importancia_depressao.png', dpi=150, bbox_inches='tight')

print("Processamento concluído. Arquivos 'metricas_desempenho_depressao.png' e 'ranking_importancia_depressao.png' gerados.")
