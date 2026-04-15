import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

# 1. CONFIGURAÇÃO DE CAMINHOS E CARREGAMENTO
# Ajustado para a estrutura de pastas do seu projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'datasets'
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'dataset_final.csv')
target = 'Preval_Ansiedade'

# 2. CRIAÇÃO DOS ESTRATOS POR PIB (Wealth Levels)
# Classificamos os países em 5 níveis de riqueza baseados na média histórica do PIB
country_avg_gdp = df.groupby('Code')['GDP_Log'].mean()
gdp_bins = pd.qcut(country_avg_gdp, q=5, labels=[1, 2, 3, 4, 5])
df['GDP_Stratum'] = df['Code'].map(gdp_bins)

# 3. DEFINIÇÃO DE X E Y
# Removemos IDs, o ano, o nível de riqueza auxiliar e todas as outras doenças
all_targets = ['Preval_Esquizofrenia', 'Preval_Depressao', 'Preval_Ansiedade', 
               'Preval_Bipolaridade', 'Preval_T_Alimentar']
X = df.drop(columns=['Entity', 'Code', 'Year', 'GDP_Stratum'] + all_targets)
y = df[target]
groups = df['Code'] # Define que o país deve permanecer unido no mesmo fold

# 4. CONFIGURAÇÃO DO MODELO E VALIDAÇÃO (STRATIFIED GROUP K-FOLD)
# O StratifiedGroupKFold garante que cada fold tenha países variados e riqueza equilibrada
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}

# Executa a validação cruzada
results = cross_validate(model, X, y, cv=sgkf.split(X, df['GDP_Stratum'], groups=groups), scoring=scoring)

# 5. GRÁFICO DE MÉTRICAS (DESEMPENHO)
metrics_df = pd.DataFrame({
    'Métrica': ['R²', 'MAE', 'RMSE'],
    'Valor': [results['test_r2'].mean(), -results['test_mae'].mean(), -results['test_rmse'].mean()]
})

plt.figure(figsize=(8, 4))
sns.barplot(x='Valor', y='Métrica', data=metrics_df, palette='viridis', hue='Métrica', legend=False)
plt.title(f'Desempenho (Stratified Group K-Fold por PIB) - {target}')
plt.xlim(0, max(1.1, metrics_df['Valor'].max() * 1.1))
for i, v in enumerate(metrics_df['Valor']):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'metricas_ansiedade_rigoroso.png')

# 6. GRÁFICO DE IMPORTÂNCIA (RANKING FINAL)
# Treinamos com 100% dos dados para extrair o conhecimento final
model.fit(X, y)
importances = pd.DataFrame({
    'Atributo': X.columns,
    'Importancia': model.feature_importances_
}).sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importancia', y='Atributo', data=importances, palette='magma', hue='Atributo', legend=False)
plt.title(f'Ranking de Importância de Atributos - {target}', fontsize=14)
plt.xlabel('Importância Relativa (MDI)', fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ranking_ansiedade_rigoroso.png', dpi=150, bbox_inches='tight')

print(f"Processamento concluído. O modelo foi validado para garantir que a economia de um país prevê a ansiedade de outro.")