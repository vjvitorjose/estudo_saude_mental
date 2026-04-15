import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from linearmodels.panel import PanelOLS
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

# Montar índice de painel (país × ano) exigido pelo linearmodels
print("Construção do painel")
df = df.set_index(['Code', 'Year'])
print(f"  Índice            : (Code, Year)")
print(f"  Entidades         : {df.index.get_level_values('Code').nunique()}")
print(f"  Períodos          : {df.index.get_level_values('Year').nunique()}\n")

X = df[FEATURE_COLS]

# Regressão com efeitos fixos por país e por ano
# entity_effects=True  → absorve características fixas de cada país (cultura, geografia, etc.)
# time_effects=True    → absorve choques globais comuns a todos os países em cada ano
# cov_type='clustered' → erros-padrão clusterizados por país, corrigindo autocorrelação temporal
print("Efeitos Fixos (Two-Way: país + ano)")
print("  entity_effects : absorve heterogeneidade fixa de cada país")
print("  time_effects   : absorve choques globais por ano")
print("  cov_type       : clustered por país (corrige autocorrelação temporal)\n")

results      = {}
fitted_vals  = {}
coef_tables  = {}

for target in TARGET_COLS:
    y = df[target]

    model  = PanelOLS(y, X, entity_effects=True, time_effects=True)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    y_true = y.values
    y_pred = result.fitted_values.values

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R² within: quanto da variação dentro de cada país é explicada
    r2_within = result.rsquared

    results[target] = {
        'r2_within' : r2_within,
        'r2_total'  : r2,
        'mae'       : mae,
        'rmse'      : rmse,
    }

    coef_tables[target] = pd.DataFrame({
        'coef'   : result.params,
        'std_err': result.std_errors,
        'pvalue' : result.pvalues,
    })

    fitted_vals[target] = (y_true, y_pred)

    label = TARGET_LABELS[target]
    print(f"  {label:<16s}:  R²_within={r2_within:.4f}"
          f"  |  R²_total={r2:.4f}"
          f"  |  MAE={mae:.6f}"
          f"  |  RMSE={rmse:.6f}")

# Tabela de coeficientes por transtorno
print("\nCoeficientes (efeitos fixos removidos)")
print("  Interpretação: variação dentro de um mesmo país ao longo do tempo.\n")

for target in TARGET_COLS:
    label = TARGET_LABELS[target]
    ct    = coef_tables[target]
    print(f"  {label}")
    print(f"  {'Feature':<15s}  {'Coef':>10s}  {'Std Err':>10s}  {'p-value':>10s}  Sig")
    for feat in FEATURE_COLS:
        coef = ct.loc[feat, 'coef']
        se   = ct.loc[feat, 'std_err']
        pv   = ct.loc[feat, 'pvalue']
        sig  = '***' if pv < 0.01 else '**' if pv < 0.05 else '*' if pv < 0.1 else ''
        print(f"  {feat:<15s}  {coef:>10.6f}  {se:>10.6f}  {pv:>10.4f}  {sig}")
    print()

# Gráfico de métricas por transtorno
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Efeitos Fixos (Two-Way) — métricas por transtorno', fontsize=12)

labels        = [TARGET_LABELS[t] for t in TARGET_COLS]
r2w_vals      = [results[t]['r2_within'] for t in TARGET_COLS]
mae_vals      = [results[t]['mae']       for t in TARGET_COLS]
rmse_vals     = [results[t]['rmse']      for t in TARGET_COLS]

bar_colors = ['#2ecc71' if v >= 0.7 else '#f39c12' if v >= 0.4 else '#e74c3c'
              for v in r2w_vals]

axes[0].barh(labels, r2w_vals, color=bar_colors, edgecolor='white', height=0.6)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('R² within')
axes[0].set_title('R² within (maior é melhor)')
axes[0].set_xlim(-0.1, 1.05)

axes[1].barh(labels, mae_vals, color='#3498db', edgecolor='white', height=0.6)
axes[1].set_xlabel('MAE')
axes[1].set_title('MAE (menor é melhor)')

axes[2].barh(labels, rmse_vals, color='#9b59b6', edgecolor='white', height=0.6)
axes[2].set_xlabel('RMSE')
axes[2].set_title('RMSE (menor é melhor)')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fe01_metricas_por_transtorno.png', dpi=150)
plt.close()
print("fe01_metricas_por_transtorno.png")

# Gráfico de coeficientes com intervalo de confiança (por transtorno)
fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(18, 5))
fig.suptitle('Coeficientes — Efeitos Fixos (Two-Way)\n(intervalo de confiança 95%)', fontsize=12)

for ax, target in zip(axes, TARGET_COLS):
    ct    = coef_tables[target]
    coefs = ct['coef']
    ci    = 1.96 * ct['std_err']
    order = np.argsort(coefs.values)
    feats = [FEATURE_COLS[i] for i in order]
    vals  = coefs.values[order]
    errs  = ci.values[order]

    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in vals]
    ax.barh(feats, vals, xerr=errs, color=colors,
            edgecolor='white', height=0.65, capsize=4)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(TARGET_LABELS[target])
    ax.set_xlabel('Coeficiente')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fe02_coeficientes.png', dpi=150)
plt.close()
print("fe02_coeficientes.png")

# Heatmap de p-values (significância estatística)
df_pval = pd.DataFrame(
    {target: coef_tables[target]['pvalue'] for target in TARGET_COLS},
    index=FEATURE_COLS
).rename(columns=TARGET_LABELS)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    df_pval,
    annot=True, fmt='.3f',
    cmap='RdYlGn_r',
    vmin=0, vmax=0.1,
    linewidths=0.5,
    ax=ax
)
ax.set_title('p-values dos coeficientes — Efeitos Fixos\n(verde = significativo, vermelho = não significativo)', fontsize=12)
ax.set_xlabel('Transtorno')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'fe03_pvalues.png', dpi=150)
plt.close()
print("fe03_pvalues.png")

# Resumo final
print("\nResumo final")
df_res = (pd.DataFrame(results).T
            .rename(index=TARGET_LABELS)
            [['r2_within', 'r2_total', 'mae', 'rmse']])
df_res.columns = ['R² within', 'R² total', 'MAE', 'RMSE']
print(df_res.to_string(float_format=lambda x: f'{x:.6f}'))
