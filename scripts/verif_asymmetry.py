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
YEAR_REF = 2010

GDP_CODE = 'NY.GDP.MKTP.CD'

WB_SERIES = {
    'SL.UEM.TOTL.NE.ZS': 'Desemprego total (%)',
    'SI.POV.GINI'       : 'Gini',
    'IT.NET.USER.ZS'    : 'Internet (% população)',
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

ref_col = f'{YEAR_REF} [YR{YEAR_REF}]'
gdp_df = df_wb[(df_wb['Series Code'] == GDP_CODE) &
                 (df_wb['Country Code'].isin(common_codes))]
gdp_ref = gdp_df[ref_col].dropna()
log_gdp = np.log(gdp_ref.replace(0, np.nan).dropna())

mh_clean = df_mh.dropna(subset=['Code'])
mh_clean = mh_clean[(mh_clean['Year'] >= YEAR_START) & (mh_clean['Year'] <= YEAR_END)]

# PIB Bruto vs transformação com log em uma snapshot de 2010
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'PIB total — distribuição bruta vs. log  (snapshot {YEAR_REF})', fontsize=12)

sns.histplot(gdp_ref / 1e9, kde=True, color='#e74c3c', ax=axes[0], bins=35)
axes[0].set_xlabel('PIB (bilhões USD)')
axes[0].set_title(f'Distribuição bruta\nskewness = {gdp_ref.skew():.2f}')

sns.histplot(log_gdp, kde=True, color='#2ecc71', ax=axes[1], bins=35)
axes[1].set_xlabel('log(PIB)')
axes[1].set_title(f'Após transformação log\nskewness = {log_gdp.skew():.2f}')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grafico04_gdp_assimetria.png', dpi=150)
plt.close()
print("grafico04_gdp_assimetria.png")

# Evolução do skewness ao longo dos anos PIB Bruto
skew_raw, skew_log = [], []
for col in target_year_cols:
    vals = gdp_df[col].dropna()
    skew_raw.append(vals.skew())
    skew_log.append(np.log(vals.replace(0, np.nan).dropna()).skew())

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(years_int, skew_raw, marker='o', color='#e74c3c', linewidth=2, label='PIB bruto')
ax.plot(years_int, skew_log, marker='s', color='#2ecc71', linewidth=2, label='log(PIB)')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Ano')
ax.set_ylabel('Skewness')
ax.set_title('Evolução da assimetria — PIB bruto vs. log(PIB), 2000–2019')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grafico05_gdp_skew_temporal.png', dpi=150)
plt.close()
print("grafico05_gdp_skew_temporal.png")

# Assimetria na snapshot de 2010 do desemprego, gini e Internet
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Distribuição das variáveis World Bank — snapshot {YEAR_REF}', fontsize=12)

for ax, (series_code, label) in zip(axes, WB_SERIES.items()):
    sub = df_wb[(df_wb['Series Code'] == series_code) &
                (df_wb['Country Code'].isin(common_codes))][ref_col].dropna()
    sns.histplot(sub, kde=True, color='#3498db', ax=ax, bins=30)
    ax.set_title(f'{label}\nskewness = {sub.skew():.2f}')
    ax.set_xlabel(label)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grafico06_distribuicoes_wb.png', dpi=150)
plt.close()
print("grafico06_distribuicoes_wb.png")

# Assimetria nos dados do Kaggle
fig, axes = plt.subplots(1, len(disorder_cols), figsize=(17, 5))
fig.suptitle('Distribuição dos transtornos mentais — Kaggle (2000–2019)', fontsize=12)

for ax, col, label in zip(axes, disorder_cols, DISORDER_LABELS.values()):
    data = mh_clean[mh_clean['Code'].isin(common_codes)][col].dropna()
    sns.histplot(data, kde=True, color='#9b59b6', ax=ax, bins=30)
    ax.set_title(f'{label}\nskewness = {data.skew():.2f}')
    ax.set_xlabel('% da população')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grafico07_distribuicoes_kaggle.png', dpi=150)
plt.close()
print("grafico07_distribuicoes_kaggle.png")
