import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / '../datasets'
PLOTS_DIR = BASE_DIR / '../plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / 'dataset_final_KNN.csv'

sns.set_theme(style="whitegrid")

FEATURE_COLS = [
    'GDP_log', 'Gini', 'Internet', 'Unemp_total', 'Urban_pct', 'Urban_growth',
]

TARGET_COLS = [
    'Preval_Esquizofrenia',
    'Preval_Depressao',
    'Preval_Ansiedade',
    'Preval_Bipolaridade',
    'Preval_T_Alimentar',
]

TARGET_LABELS = {
    'Preval_Esquizofrenia' : 'Esquizofrenia',
    'Preval_Depressao'     : 'Depressão',
    'Preval_Ansiedade'     : 'Ansiedade',
    'Preval_Bipolaridade'  : 'Bipolaridade',
    'Preval_T_Alimentar'   : 'T. Alimentares',
}

RF_PARAMS = dict(
    n_estimators    = 1000,
    max_depth       = None,
    min_samples_leaf= 8,
    max_features    = 'sqrt',
    random_state    = 42,
    n_jobs          = -1,
)

N_SPLITS = 10

# Carregar dados
df = pd.read_csv(INPUT_FILE)
print("CARGA")
print(f"Shape             : {df.shape}")
print(f"Países            : {df['Code'].nunique()}")
print(f"Período           : {df['Year'].min()}–{df['Year'].max()}\n")

# Remover linhas com NaN nas features
print("Tratamento de NaN")
n_antes = len(df)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
n_depois = len(df)
print(f"  Linhas removidas  : {n_antes - n_depois} (NaN em features)")
print(f"  Shape após limpeza: {df.shape}")
print(f"  Países restantes  : {df['Code'].nunique()}\n")

X      = df[FEATURE_COLS].copy()
groups = df['Code'].values

print(f"Features          : {FEATURE_COLS}")
print(f"Targets           : {TARGET_COLS}")
print(f"NaN em X          : {X.isnull().sum().sum()}\n")

# Group K-Fold estratificado
print(f"Validação cruzada (StratifiedGroupKFold, k={N_SPLITS})")
print("Países agrupados e estratificados por quantis de GDP_log.")

# Balancear com base no GDP transformado com log
country_gdp = df.groupby('Code')['GDP_log'].mean()
gdp_bins    = pd.qcut(country_gdp, q=N_SPLITS, labels=False, duplicates='drop')
df['GDP_Stratum'] = df['Code'].map(gdp_bins)

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cv_folds = list(sgkf.split(X, df['GDP_Stratum'], groups))

results = {}

for target in TARGET_COLS:
    y  = df[target].values
    rf = RandomForestRegressor(**RF_PARAMS)

    r2_scores   = cross_val_score(rf, X, y, cv=cv_folds, scoring='r2')
    mae_scores  = -cross_val_score(rf, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(rf, X, y, cv=cv_folds, scoring='neg_mean_squared_error'))

    results[target] = {
        'r2_mean'   : r2_scores.mean(),
        'r2_std'    : r2_scores.std(),
        'mae_mean'  : mae_scores.mean(),
        'mae_std'   : mae_scores.std(),
        'rmse_mean' : rmse_scores.mean(),
        'rmse_std'  : rmse_scores.std(),
    }

    label = TARGET_LABELS[target]
    print(f"  {label:<16s}:  R²={r2_scores.mean():.4f} ± {r2_scores.std():.4f}"
          f"  |  MAE={mae_scores.mean():.6f} ± {mae_scores.std():.6f}"
          f"  |  RMSE={rmse_scores.mean():.6f} ± {rmse_scores.std():.6f}")

# Treinar modelos finais para importância de features
print("\nTreinando modelos finais para importância de features")

importances_mdi  = {}
importances_perm = {}

for target in TARGET_COLS:
    y  = df[target].values
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X, y)

    importances_mdi[target] = rf.feature_importances_

    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    importances_perm[target] = perm.importances_mean

    print(f"  {TARGET_LABELS[target]:<16s}: treinado")

# Gráfico de métricas por transtorno
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Random Forest — métricas por transtorno\n(GroupKFold por país, k={N_SPLITS})', fontsize=12)

labels     = [TARGET_LABELS[t] for t in TARGET_COLS]
r2_means   = [results[t]['r2_mean']   for t in TARGET_COLS]
mae_means  = [results[t]['mae_mean']  for t in TARGET_COLS]
rmse_means = [results[t]['rmse_mean'] for t in TARGET_COLS]
r2_stds    = [results[t]['r2_std']    for t in TARGET_COLS]
mae_stds   = [results[t]['mae_std']   for t in TARGET_COLS]
rmse_stds  = [results[t]['rmse_std']  for t in TARGET_COLS]

bar_colors = ['#2ecc71' if v >= 0.7 else '#f39c12' if v >= 0.4 else '#e74c3c'
              for v in r2_means]

axes[0].barh(labels, r2_means, xerr=r2_stds, color=bar_colors,
             edgecolor='white', height=0.6, capsize=4)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('R²')
axes[0].set_title('R²')
axes[0].set_xlim(-0.1, 1.05)

axes[1].barh(labels, mae_means, xerr=mae_stds, color='#3498db',
             edgecolor='white', height=0.6, capsize=4)
axes[1].set_xlabel('MAE')
axes[1].set_title('MAE')

axes[2].barh(labels, rmse_means, xerr=rmse_stds, color='#9b59b6',
             edgecolor='white', height=0.6, capsize=4)
axes[2].set_xlabel('RMSE')
axes[2].set_title('RMSE')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rf01_metricas_por_transtorno.png', dpi=150)
plt.close()
print("\nrf01_metricas_por_transtorno.png")

# Gráfico de importância MDI por transtorno
fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(18, 5))
fig.suptitle('Importância das features (MDI) — Random Forest', fontsize=12)

for ax, target in zip(axes, TARGET_COLS):
    imp   = importances_mdi[target]
    order = np.argsort(imp)
    ax.barh([FEATURE_COLS[i] for i in order], imp[order],
            color='#2ecc71', edgecolor='white', height=0.65)
    ax.set_title(TARGET_LABELS[target])
    ax.set_xlabel('Importância (MDI)')
    ax.set_xlim(0, imp.max() * 1.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rf02_importancia_mdi.png', dpi=150)
plt.close()
print("rf02_importancia_mdi.png")

# Gráfico de importância por permutação (heatmap — todos os transtornos)
df_perm = pd.DataFrame(
    importances_perm,
    index=FEATURE_COLS
).rename(columns=TARGET_LABELS)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    df_perm,
    annot=True, fmt='.4f',
    cmap='YlGn',
    linewidths=0.5,
    ax=ax
)
ax.set_title('Importância por permutação — Random Forest\n(média sobre 20 repetições)', fontsize=12)
ax.set_xlabel('Transtorno')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rf03_importancia_permutacao.png', dpi=150)
plt.close()
print("rf03_importancia_permutacao.png")

# Resumo final
print("\nResumo final")
df_res = (pd.DataFrame(results).T
            .rename(index=TARGET_LABELS)
            [['r2_mean', 'r2_std', 'mae_mean', 'rmse_mean']])
df_res.columns = ['R² médio', 'R² std', 'MAE médio', 'RMSE médio']
print(df_res.to_string(float_format=lambda x: f'{x:.6f}'))
