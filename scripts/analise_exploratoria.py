import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'datasets'
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

def clean_wb_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Country Code']) # Remove rodapés
    df = df.replace('..', np.nan) # Converte símbolos em nulos
    
    # Converte anos para numérico
    year_cols = [c for c in df.columns if '[' in c]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df, year_cols

df_wb, wb_years = clean_wb_data(DATA_DIR / 'ihme_original.csv')
df_mh = pd.read_csv(DATA_DIR / 'kaggle_original.csv')

disorder_cols = [c for c in df_mh.columns if 'share' in c]
short_names = ['Esquizofrenia', 'Depressão', 'Ansiedade', 'Bipolaridade', 'T. Alimentares']
mh_rename = dict(zip(disorder_cols, short_names))

# --- PLOTS ---

# 1. Distribuição dos Transtornos Mentais
plt.figure(figsize=(15, 10))
for i, col in enumerate(disorder_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(df_mh[col], kde=True, color='skyblue')
    plt.title(f'Distribuição: {mh_rename[col]}')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_mh_distributions.png')

# 2. Distribuição Socioeconômica (Exemplo: 2015)
key_series = {'NY.GDP.MKTP.CD': 'PIB', 'SL.UEM.TOTL.NE.ZS': 'Desemprego', 'SI.POV.GINI': 'Gini'}
plt.figure(figsize=(15, 5))
for i, (code, name) in enumerate(key_series.items()):
    plt.subplot(1, 3, i+1)
    data = df_wb[df_wb['Series Code'] == code]['2015 [YR2015]'].dropna()
    sns.histplot(data, kde=True, color='salmon')
    plt.title(f'Distribuição: {name}')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_wb_distributions.png')

# 3. Matriz de Correlação (Saúde Mental)
plt.figure(figsize=(10, 8))
sns.heatmap(df_mh[disorder_cols].rename(columns=mh_rename).corr(), annot=True, cmap='coolwarm')
plt.title('Correlação entre Transtornos')
plt.savefig(PLOTS_DIR / 'eda_mh_correlation.png')

# 4. Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_mh[disorder_cols].rename(columns=mh_rename))
plt.title('Detecção de Outliers: Saúde Mental')
plt.savefig(PLOTS_DIR / 'eda_outliers.png')

print("EDA Concluída. Gráficos salvos.")