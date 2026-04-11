import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / '../datasets'
PLOTS_DIR = BASE_DIR / '../plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

YEAR_START = 2000
YEAR_END = 2019

LOW_COVERAGE_SERIES = {
    'GC.DOD.TOTL.GD.ZS': 'Dívida pública (% PIB)',
    'SI.POV.GINI'       : 'Gini',
    'SL.UEM.ADVN.ZS'    : 'Desemprego (ens. superior)',
    'SL.UEM.BASC.ZS'    : 'Desemprego (ens. básico)',
    'SL.UEM.INTM.ZS'    : 'Desemprego (ens. médio)',
    'SL.UEM.TOTL.NE.ZS' : 'Desemprego total',
}

# Carregar dados
def clean_wb_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Country Code'])
    df = df.replace('..', np.nan)
    year_cols = [c for c in df.columns if '[' in c]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df, year_cols

df_wb, wb_year_cols = clean_wb_data(DATA_DIR / 'ihme_original.csv')
df_mh = pd.read_csv(DATA_DIR / 'kaggle_original.csv')

target_year_cols = [c for c in wb_year_cols if YEAR_START <= int(c[:4]) <= YEAR_END]
years_int = [int(c[:4]) for c in target_year_cols]
n_years = len(target_year_cols)

wb_codes = set(df_wb['Country Code'].dropna().unique())
kaggle_codes = set(df_mh['Code'].dropna().unique())
common_codes = wb_codes & kaggle_codes
n_countries = len(common_codes)

def max_consecutive_gap(series: pd.Series) -> int:
    max_gap = current = 0
    for v in series.values:
        if pd.isna(v):
            current += 1
            max_gap  = max(max_gap, current)
        else:
            current  = 0
    return max_gap

# Histograma + cobertura anual por série
for series_code, label in LOW_COVERAGE_SERIES.items():

    sub = df_wb[(df_wb['Series Code'] == series_code) &
                          (df_wb['Country Code'].isin(common_codes))].copy()
    country_pivot = sub.set_index('Country Code')[target_year_cols]

    overall_cov = country_pivot.notna().sum().sum() / (n_countries * n_years) * 100
    gap_values = country_pivot.apply(max_consecutive_gap, axis=1)
    gap_counts = gap_values.value_counts().sort_index()
    annual_cov = [country_pivot[c].notna().sum() / n_countries * 100
                     for c in target_year_cols]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{label}  —  cobertura geral: {overall_cov:.1f}%\n'
                 f'({n_countries} países comuns, 2000–{YEAR_END})', fontsize=12)

    # Esquerda: histograma de gaps
    bar_colors = ['#2ecc71' if i == 0 else '#f39c12' if i <= 5
                  else '#e67e22' if i <= 10 else '#e74c3c'
                  for i in gap_counts.index]
    axes[0].bar(gap_counts.index, gap_counts.values,
                color=bar_colors, edgecolor='white')
    axes[0].set_xlabel('Maior gap consecutivo (anos sem dado)')
    axes[0].set_ylabel('Número de países')
    axes[0].set_title('Distribuição de gaps por país')

    # Direita: cobertura por ano
    axes[1].plot(years_int, annual_cov, marker='o', color='#3498db',
                 linewidth=2, markersize=5)
    axes[1].fill_between(years_int, annual_cov, alpha=0.15, color='#3498db')
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel('Ano')
    axes[1].set_ylabel('% países com dado')
    axes[1].set_title('Cobertura por ano')

    plt.tight_layout()
    fname = f"grafico03_gaps_{series_code.replace('.', '_')}.png"
    plt.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close()
    print(f"{fname}")
