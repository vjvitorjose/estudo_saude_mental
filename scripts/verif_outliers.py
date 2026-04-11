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

GDP_CODE = 'NY.GDP.MKTP.CD'

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

wb_codes = set(df_wb['Country Code'].dropna().unique())
kaggle_codes = set(df_mh['Code'].dropna().unique())
common_codes = wb_codes & kaggle_codes

gdp_df = df_wb[(df_wb['Series Code'] == GDP_CODE) &
               (df_wb['Country Code'].isin(common_codes))].copy()

gdp_long = (gdp_df[['Country Code'] + target_year_cols]
            .melt(id_vars='Country Code', var_name='year_str', value_name='gdp_raw')
            .assign(year=lambda d: d['year_str'].str[:4].astype(int))
            .drop(columns='year_str')
            .dropna())
gdp_long['log_gdp'] = np.log(gdp_long['gdp_raw'].replace(0, np.nan))

# GDP boxplots por ano bruto x log
gdp_pivot = gdp_long.pivot_table(index='year', columns='Country Code', values='gdp_raw')
log_gdp_pivot = gdp_long.pivot_table(index='year', columns='Country Code', values='log_gdp')

year_range = list(range(YEAR_START, YEAR_END + 1))
raw_by_year = [gdp_pivot.loc[yr].dropna().values / 1e12 for yr in year_range]
log_by_year = [log_gdp_pivot.loc[yr].dropna().values   for yr in year_range]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Outliers — PIB total por ano (2000–2019)', fontsize=12)

axes[0].boxplot(raw_by_year, tick_labels=year_range, patch_artist=True,
                boxprops=dict(facecolor='#fadbd8'),
                flierprops=dict(marker='.', markersize=4, color='#e74c3c'))
axes[0].set_xticklabels(year_range, rotation=45, fontsize=8)
axes[0].set_ylabel('PIB (trilhões USD)')
axes[0].set_title('PIB bruto')

axes[1].boxplot(log_by_year, tick_labels=year_range, patch_artist=True,
                boxprops=dict(facecolor='#d5f5e3'),
                flierprops=dict(marker='.', markersize=4, color='#27ae60'))
axes[1].set_xticklabels(year_range, rotation=45, fontsize=8)
axes[1].set_ylabel('log(PIB)')
axes[1].set_title('log(PIB)')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'grafico08_outliers_gdp.png', dpi=150)
plt.close()
print("grafico08_outliers_gdp.png")
