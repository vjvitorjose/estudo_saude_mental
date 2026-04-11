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

SERIES_LABELS = {
    'NY.GDP.MKTP.CD'   : 'PIB total',
    'IT.NET.USER.ZS'   : 'Internet',
    'SL.UEM.ADVN.ZS'   : 'Desemprego (ens. superior)',
    'SL.UEM.BASC.ZS'   : 'Desemprego (ens. básico)',
    'SL.UEM.INTM.ZS'   : 'Desemprego (ens. médio)',
    'SL.UEM.TOTL.NE.ZS': 'Desemprego total',
    'SP.URB.TOTL.IN.ZS': 'Urbanização (%)',
    'SP.URB.GROW'      : 'Crescimento urbano (%)',
    'SI.POV.GINI'      : 'Gini',
    'GC.DOD.TOTL.GD.ZS': 'Dívida pública (% PIB)',
}

DISORDER_LABELS = {
    'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Esquizofrenia',
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized'   : 'Depressão',
    'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized'      : 'Ansiedade',
    'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized'      : 'Bipolaridade',
    'Eating disorders (share of population) - Sex: Both - Age: Age-standardized'       : 'T. Alimentares',
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
disorder_cols = list(DISORDER_LABELS.keys())

target_year_cols = [c for c in wb_year_cols if YEAR_START <= int(c[:4]) <= YEAR_END]
years_int = [int(c[:4]) for c in target_year_cols]

wb_codes = set(df_wb['Country Code'].dropna().unique())
kaggle_codes = set(df_mh['Code'].dropna().unique())
common_codes = wb_codes & kaggle_codes
n_countries = len(common_codes)

mh_clean = df_mh.dropna(subset=['Code'])
mh_clean = mh_clean[(mh_clean['Year'] >= YEAR_START) & (mh_clean['Year'] <= YEAR_END)]

# Cobertura geral por série
wb_coverage = {}
for series_code, label in SERIES_LABELS.items():
    sub = df_wb[(df_wb['Series Code'] == series_code) &
                         (df_wb['Country Code'].isin(common_codes))]
    total_cells = n_countries * len(target_year_cols)
    filled_cells = sub[target_year_cols].notna().sum().sum()
    wb_coverage[label] = round(filled_cells / total_cells * 100, 1)

mh_coverage = {}
for col, label in DISORDER_LABELS.items():
    sub = mh_clean[mh_clean['Code'].isin(common_codes)]
    total_cells = n_countries * len(target_year_cols)
    filled_cells = sub[col].notna().sum()
    mh_coverage[label] = round(filled_cells / total_cells * 100, 1)

# Cobertura geral por atributo
all_labels = list(wb_coverage.keys()) + list(mh_coverage.keys())
all_values = list(wb_coverage.values()) + list(mh_coverage.values())
all_sources = ['World Bank'] * len(wb_coverage) + ['Kaggle (GBD)'] * len(mh_coverage)

df_bar = (pd.DataFrame({'variavel': all_labels,
                        'cobertura': all_values,
                        'fonte': all_sources})
            .sort_values('cobertura'))

bar_colors = ['#2ecc71' if v >= 70 else '#f39c12' if v >= 30 else '#e74c3c'
              for v in df_bar['cobertura']]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(df_bar['variavel'], df_bar['cobertura'],
               color=bar_colors, edgecolor='white', height=0.65)

for bar, val, source in zip(bars, df_bar['cobertura'], df_bar['fonte']):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}%  [{source}]',
            va='center', fontsize=9, color='#555')

from matplotlib.patches import Patch
legend_patches = [
    Patch(color='#2ecc71', label='≥ 70%'),
    Patch(color='#f39c12', label='30–70%'),
    Patch(color='#e74c3c', label='< 30%'),
]
ax.legend(handles=legend_patches, fontsize=9, loc='lower right')
ax.set_xlim(0, 120)
ax.set_xlabel('Cobertura (% entradas com dado — 200 países, 2000–2019)')
ax.set_title('Cobertura de dados por variável\nWorld Bank + Kaggle (GBD)', fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grafico01_cobertura_geral.png', dpi=150)
plt.close()
print("grafico01_cobertura_geral.png")

# Cobertura anual por atributo
fig, ax = plt.subplots(figsize=(13, 6))

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(SERIES_LABELS))]

for (series_code, label), color in zip(SERIES_LABELS.items(), colors):
    sub = df_wb[(df_wb['Series Code'] == series_code) &
                (df_wb['Country Code'].isin(common_codes))]
    annual_cov = [sub[c].notna().sum() / n_countries * 100 for c in target_year_cols]
    ax.plot(years_int, annual_cov, marker='o', markersize=4,
            linewidth=1.8, label=label, color=color)

ax.set_ylim(0, 105)
ax.set_xlabel('Ano')
ax.set_ylabel('% países com dado')
ax.set_title('Cobertura anual por variável — World Bank\n(200 países comuns, 2000–2019)', fontsize=11)
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grafico02_cobertura_anual_wb.png', dpi=150, bbox_inches='tight')
plt.close()
print("grafico02_cobertura_anual_wb.png")
