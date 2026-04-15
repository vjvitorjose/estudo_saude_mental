import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
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
    n_estimators     = 1000,
    max_depth        = None,
    min_samples_leaf = 8,
    max_features     = 'sqrt',
    n_jobs           = -1,
)

N_SPLITS     = 10
N_REPEATS    = 10
PERM_REPEATS = 20

# Carregar dados
df = pd.read_csv(INPUT_FILE)
print("CARGA")
print(f"  Shape             : {df.shape}")
print(f"  Países            : {df['Code'].nunique()}")
print(f"  Período           : {df['Year'].min()}–{df['Year'].max()}")

n_antes = len(df)
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"  Linhas removidas  : {n_antes - len(df)} (NaN em features)")
print(f"  Shape após limpeza: {df.shape}")
print(f"  Países restantes  : {df['Code'].nunique()}\n")

X      = df[FEATURE_COLS].values
groups = df['Code'].values

# Estratificação por GDP médio por país
country_gdp      = df.groupby('Code')['GDP_log'].mean()
gdp_bins         = pd.qcut(country_gdp, q=N_SPLITS, labels=False, duplicates='drop')
df['GDP_Stratum'] = df['Code'].map(gdp_bins)
strata           = df['GDP_Stratum'].values

# Validação cruzada repetida com seeds diferentes
print(f"Validação cruzada repetida: {N_REPEATS} repetições × {N_SPLITS} folds")
print(f"Estratégia: StratifiedGroupKFold — países inteiros, estratificados por GDP\n")

all_scores = {t: {'r2': [], 'mae': [], 'rmse': []} for t in TARGET_COLS}

for rep in range(N_REPEATS):
    seed = 42 + rep
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(sgkf.split(X, strata, groups))

    for target in TARGET_COLS:
        y  = df[target].values
        rf = RandomForestRegressor(**RF_PARAMS, random_state=seed)

        r2   = cross_val_score(rf, X, y, cv=folds, scoring='r2')
        mae  = -cross_val_score(rf, X, y, cv=folds, scoring='neg_mean_absolute_error')
        rmse = np.sqrt(-cross_val_score(rf, X, y, cv=folds,
                                        scoring='neg_mean_squared_error'))

        all_scores[target]['r2'].extend(r2.tolist())
        all_scores[target]['mae'].extend(mae.tolist())
        all_scores[target]['rmse'].extend(rmse.tolist())

    print(f"  Repetição {rep+1:2d}/{N_REPEATS} concluída")

# Agregar: média e desvio padrão sobre todos os folds de todas as repetições
results = {}
for target in TARGET_COLS:
    results[target] = {
        'r2_mean'  : np.mean(all_scores[target]['r2']),
        'r2_std'   : np.std(all_scores[target]['r2']),
        'mae_mean' : np.mean(all_scores[target]['mae']),
        'mae_std'  : np.std(all_scores[target]['mae']),
        'rmse_mean': np.mean(all_scores[target]['rmse']),
        'rmse_std' : np.std(all_scores[target]['rmse']),
    }

print("\nResumo de métricas (média ± desvio padrão sobre 100 avaliações):")
for target, m in results.items():
    print(f"  {TARGET_LABELS[target]:<16s}:  "
          f"R²={m['r2_mean']:.4f} ± {m['r2_std']:.4f}  |  "
          f"MAE={m['mae_mean']:.6f} ± {m['mae_std']:.6f}  |  "
          f"RMSE={m['rmse_mean']:.6f} ± {m['rmse_std']:.6f}")

# Média da importância de features
print(f"\nImportância de features ({N_REPEATS} modelos finais por transtorno)")

mdi_accum  = {t: np.zeros(len(FEATURE_COLS)) for t in TARGET_COLS}
perm_accum = {t: np.zeros(len(FEATURE_COLS)) for t in TARGET_COLS}

for rep in range(N_REPEATS):
    seed = 42 + rep
    for target in TARGET_COLS:
        y  = df[target].values
        rf = RandomForestRegressor(**RF_PARAMS, random_state=seed)
        rf.fit(X, y)

        mdi_accum[target] += rf.feature_importances_

        perm = permutation_importance(rf, X, y,
                                      n_repeats=PERM_REPEATS,
                                      random_state=seed,
                                      n_jobs=-1)
        perm_accum[target] += perm.importances_mean

    print(f"  Repetição {rep+1:2d}/{N_REPEATS} concluída")

# Média sobre as N_REPEATS repetições
importances_mdi  = {t: mdi_accum[t]  / N_REPEATS for t in TARGET_COLS}
importances_perm = {t: perm_accum[t] / N_REPEATS for t in TARGET_COLS}

# Métricas por transtorno
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f'Random Forest — métricas por transtorno\n'
    f'(GroupKFold por país, {N_REPEATS} repetições × {N_SPLITS} folds = {N_REPEATS*N_SPLITS} avaliações)',
    fontsize=11
)

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
axes[0].set_xlim(min(r2_means) - 0.15, 1.05)

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

# Importância MDI por transtorno
fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(18, 5))
fig.suptitle(
    f'Importância das features (MDI) — Random Forest\n'
    f'(média de {N_REPEATS} modelos)',
    fontsize=12
)

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

# Importância por permutação (heatmap)
df_perm = pd.DataFrame(importances_perm, index=FEATURE_COLS).rename(columns=TARGET_LABELS)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df_perm, annot=True, fmt='.4f', cmap='YlGn',
            linewidths=0.5, ax=ax)
ax.set_title(
    f'Importância por permutação — Random Forest\n'
    f'(média de {N_REPEATS} modelos × {PERM_REPEATS} permutações cada)',
    fontsize=11
)
ax.set_xlabel('Transtorno')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rf03_importancia_permutacao.png', dpi=150)
plt.close()
print("rf03_importancia_permutacao.png")

# Boxplot da distribuição de R² por transtorno (todas as avaliações)
r2_data = {TARGET_LABELS[t]: all_scores[t]['r2'] for t in TARGET_COLS}
df_box  = pd.DataFrame(r2_data)

fig, ax = plt.subplots(figsize=(10, 5))
df_box.boxplot(ax=ax, patch_artist=True,
               boxprops=dict(facecolor='#d6eaf8', color='#2980b9'),
               medianprops=dict(color='#e74c3c', linewidth=2),
               whiskerprops=dict(color='#2980b9'),
               capprops=dict(color='#2980b9'),
               flierprops=dict(marker='.', markersize=4, color='#7f8c8d'))
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel('R²')
ax.set_title(
    f'Distribuição do R² por transtorno\n'
    f'({N_REPEATS} repetições × {N_SPLITS} folds = {N_REPEATS*N_SPLITS} avaliações por transtorno)',
    fontsize=11
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rf04_r2_distribuicao.png', dpi=150)
plt.close()
print("rf04_r2_distribuicao.png")

# Resumo Final
print("\nResumo final")
df_res = (pd.DataFrame(results).T
            .rename(index=TARGET_LABELS)
            [['r2_mean', 'r2_std', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']])
df_res.columns = ['R² médio', 'R² std', 'MAE médio', 'MAE std', 'RMSE médio', 'RMSE std']
print(df_res.to_string(float_format=lambda x: f'{x:.6f}'))
