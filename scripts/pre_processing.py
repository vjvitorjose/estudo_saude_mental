import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / '../datasets'
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE  = DATA_DIR / 'merged_raw.csv'
OUTPUT_MAIN = DATA_DIR / 'dataset_preprocessado.csv'
OUTPUT_BAL  = DATA_DIR / 'dataset_final_KNN.csv'

YEAR_START = 2000
YEAR_END   = 2019

RENAME_MAP = {
    'GDP (current US$)' : 'GDP_raw',
    'Gini index' : 'Gini',
    'Individuals using the Internet (% of population)' : 'Internet',
    'Unemployment, total (% of total labor force) (national estimate)' : 'Unemp_total',
    'Urban population (% of total population)' : 'Urban_pct',
    'Urban population growth (annual %)' : 'Urban_growth',
}

COLS_TO_DROP = [
    'Central government debt, total (% of GDP)',
    'Unemployment with advanced education (% of total labor force with advanced education)',
    'Unemployment with basic education (% of total labor force with basic education)',
    'Unemployment with intermediate education (% of total labor force with intermediate education)',
]

# Variáveis usadas como referência de similaridade no KNN
KNN_REFERENCE_COLS = [
    'GDP_log', 'Urban_pct', 'Urban_growth', 'Internet',
    'Preval_Depressao', 'Preval_Ansiedade',
]

# Colunas alvo do KNN
KNN_TARGET_COLS = ['Gini', 'Unemp_total']

# Colunas tratadas com interpolação linear
INTERP_COLS = {
    'Internet': 3,
    'GDP_log' : 2,
}

# Colunas a serem padronizadas
ECON_COLS_TO_STANDARDIZE = [
    'GDP_log', 'Gini', 'Internet', 'Unemp_total', 'Urban_pct', 'Urban_growth',
]

# Carregar dados
df = pd.read_csv(INPUT_FILE)
print(f"CARGA")
print(f"Shape original  : {df.shape}")
print(f"Países          : {df['Code'].nunique()}")
print(f"Período         : {df['Year'].min()}–{df['Year'].max()}\n")

# Corte do período
print(f"\nCorte de período: {YEAR_START}–{YEAR_END}")

n_antes = len(df)
df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)].copy()
print(f"Linhas removidas: {n_antes - len(df)} (anos 1993–1999)")
print(f"Shape após corte: {df.shape}\n")

# Descartar variáveis com cobertura muito baixa
print(f"Descartando variáveis de baixa cobertura")

for col in COLS_TO_DROP:
    if col in df.columns:
        cob = (1 - df[col].isnull().sum() / len(df)) * 100
        print(f"  Removida : {col[:65]}")
        print(f"             cobertura = {cob:.1f}%")
        df = df.drop(columns=[col])

print("\nColunas após renomeação")
df = df.rename(columns=RENAME_MAP)
for col in df.columns:
    print(f"  {col}")

# Transformação log no GDP
print("\nTransformação log no GDP")
skew_raw   = df['GDP_raw'].skew()
df['GDP_log'] = np.log(df['GDP_raw'].replace(0, np.nan))
skew_log   = df['GDP_log'].skew()
print(f"Skewness GDP bruto : {skew_raw:.2f}")
print(f"Skewness log(GDP)  : {skew_log:.2f}\n")

# Flags de imputação
print(f"\nFlags de imputação")

for col in KNN_TARGET_COLS + list(INTERP_COLS.keys()):
    df[f'{col}_imputed']        = df[col].isnull()
    df[f'{col}_imputed_method'] = np.where(df[col].isnull(), '', 'original')
    print(f"  {col}: {df[f'{col}_imputed'].sum()} valores faltantes marcados")

print("\nInterpolação linear (colunas com cobertura >= 70%)")
df = df.sort_values(['Code', 'Year']).reset_index(drop=True)

for col, limit in INTERP_COLS.items():
    n_antes = df[col].isnull().sum()
    df[col] = (df.groupby('Code')[col]
                 .transform(lambda s: s.interpolate(
                     method='linear', limit=limit, limit_direction='both')))
    n_depois  = df[col].isnull().sum()
    imputados = n_antes - n_depois
    mask_interp = df[f'{col}_imputed'] & df[col].notna()
    df.loc[mask_interp, f'{col}_imputed_method'] = 'interpolacao_linear'
    print(f"  {col:<12s}: limit={limit}  |  {imputados:4d} interpolados  |  {n_depois:3d} NaN restantes")

# Imputar dados com KNN
print("\nImputando dados com KNN (Gini e Unemp_total)")
print("  Lógica: para cada observação (país × ano) com valor faltante,")
print("  o KNN encontra os K=5 países mais similares NO MESMO ANO,")
print("  usando as variáveis de referência como base de distância.")

knn = KNNImputer(n_neighbors=5, weights='distance')
years = sorted(df['Year'].unique())
n_knn_total = {col: 0 for col in KNN_TARGET_COLS}

for year in years:
    mask_year = df['Year'] == year
    cols_knn  = KNN_REFERENCE_COLS + KNN_TARGET_COLS
    cols_knn  = [c for c in cols_knn if c in df.columns]
    subset    = df.loc[mask_year, cols_knn].copy()

    was_null  = {col: subset[col].isnull().copy() for col in KNN_TARGET_COLS}

    n_completos = subset[KNN_REFERENCE_COLS].dropna().shape[0]
    if n_completos >= 5:
        imputed = knn.fit_transform(subset)
        df_imp  = pd.DataFrame(imputed, columns=cols_knn, index=subset.index)
        
        for col in KNN_TARGET_COLS:
            filled = was_null[col] & df_imp[col].notna()
            df.loc[mask_year & was_null[col], col] = df_imp.loc[was_null[col], col]
            n_knn_total[col] += filled.sum()
            df.loc[mask_year & filled, f'{col}_imputed_method'] = 'knn'

for col in KNN_TARGET_COLS:
    n_restantes = df[col].isnull().sum()
    print(f"  {col:<12s}: {n_knn_total[col]:4d} imputados via KNN  |  {n_restantes:3d} NaN restantes")

# Atualizar flag final
for col in KNN_TARGET_COLS + list(INTERP_COLS.keys()):
    df[f'{col}_imputed'] = df[f'{col}_imputed'] & df[col].notna()

print("\nPadronização com z-score nas variáveis econômicas")
print("Prevalências e GDP_raw NÃO padronizados")

for col in ECON_COLS_TO_STANDARDIZE:
    if col in df.columns:
        mean = df[col].mean()
        std  = df[col].std()
        df[col] = (df[col] - mean) / std
        print(f"  {col:<15s}: mean={mean:.4f}  std={std:.4f}")

# Organizar colunas
print("\nColunas finais")
id_cols = ['Entity', 'Code', 'Year']
prev_cols = ['Preval_Esquizofrenia', 'Preval_Depressao', 'Preval_Ansiedade',
             'Preval_Bipolaridade', 'Preval_T_Alimentar']
econ_cols = ['GDP_log', 'GDP_raw', 'Gini', 'Internet',
             'Unemp_total', 'Urban_pct', 'Urban_growth']
flag_cols = [f'{c}_imputed' for c in KNN_TARGET_COLS + list(INTERP_COLS.keys())]
meth_cols = [f'{c}_imputed_method' for c in KNN_TARGET_COLS + list(INTERP_COLS.keys())]

final_cols = [c for c in id_cols + prev_cols + econ_cols + flag_cols + meth_cols
              if c in df.columns]
df = df[final_cols]

for c in final_cols:
    print(f"  {c}")

# Painel balanceado
print("\nPainel balanceado")

vars_modelo = prev_cols + ['GDP_log', 'Gini', 'Unemp_total', 'Urban_pct']
df_completo = df.dropna(subset=vars_modelo)
n_anos      = YEAR_END - YEAR_START + 1
paises_bal  = (df_completo.groupby('Code')['Year']
               .count()
               .pipe(lambda s: s[s == n_anos])
               .index)
df_bal = df_completo[df_completo['Code'].isin(paises_bal)].copy()

print(f"Painel principal  : {df['Code'].nunique()} países, {len(df)} obs")
print(f"Painel balanceado : {df_bal['Code'].nunique()} países, {len(df_bal)} obs\n")

# Nulos restantes
print("Nulos restantes no painel principal")
nulos = df[econ_cols].isnull().sum()
nulos = nulos[nulos > 0]
if len(nulos) == 0:
    print("  Nenhum NaN.")
else:
    for col, n in nulos.items():
        print(f"  {col}: {n} NaN ({n/len(df)*100:.1f}%)")

# Resumo de imputação
print("\nResumo final")
for col in KNN_TARGET_COLS + list(INTERP_COLS.keys()):
    meth_col = f'{col}_imputed_method'
    if meth_col in df.columns:
        counts = df[meth_col].replace('', 'original').value_counts()
        print(f"  {col}:")
        for method, count in counts.items():
            print(f"    {method:<25s}: {count}")

# Exportar csv
df.to_csv(OUTPUT_MAIN, index=False)
df_bal.to_csv(OUTPUT_BAL, index=False)

print(f"\n{OUTPUT_MAIN.name} → {len(df)} observações, {df['Code'].nunique()} países")
print(f"{OUTPUT_BAL.name} → {len(df_bal)} observações, {df_bal['Code'].nunique()} países")
