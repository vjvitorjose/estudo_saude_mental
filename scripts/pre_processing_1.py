import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregamento dos Dados Originais
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'datasets')

df_wb_raw = pd.read_csv(os.path.join(data_dir, 'ihme_original.csv'))
df_mh_raw = pd.read_csv(os.path.join(data_dir, 'kaggle_original.csv'))

# --- 2. Preparação do Banco Mundial (Wide to Long) ---
# Remove rodapés e converte '..' em NaN
df_wb = df_wb_raw.dropna(subset=['Country Code']).replace('..', np.nan).copy()
year_cols = [col for col in df_wb.columns if '[' in col]
id_vars = ['Country Name', 'Country Code', 'Series Name', 'Series Code']

# Derrete o dataframe (melt) e limpa a coluna de Ano
df_wb_long = pd.melt(df_wb, id_vars=id_vars, value_vars=year_cols, var_name='Year_Raw', value_name='Value')
df_wb_long['Year'] = df_wb_long['Year_Raw'].str.extract(r'(\d{4})').astype(int)
df_wb_long['Value'] = pd.to_numeric(df_wb_long['Value'], errors='coerce')

# Remove a coluna Year_Raw após extrair o ano
df_wb_long = df_wb_long.drop(columns=['Year_Raw'])

output_path = os.path.join(data_dir, 'ihme_melted.csv')
df_wb_long.to_csv(output_path, index=False)
print(f"Dataset '{output_path}' gerado com sucesso!")

# Pivota os indicadores para se tornarem colunas individuais
df_wb_pivoted = df_wb_long.pivot_table(index=['Country Name', 'Country Code', 'Year'], 
                                       columns='Series Name', 
                                       values='Value').reset_index()

output_path = os.path.join(data_dir, 'ihme_pivoted.csv')
df_wb_pivoted.to_csv(output_path, index=False)
print(f"Dataset '{output_path}' gerado com sucesso!")

# --- 3. Preparação da Saúde Mental ---
mh_rename = {
    'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Preval_Esquizofrenia',
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Preval_Depressao',
    'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Preval_Ansiedade',
    'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Preval_Bipolaridade',
    'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Preval_T_Alimentar'
}
df_mh = df_mh_raw.rename(columns=mh_rename)

# --- 4. Merge (União dos Datasets) ---
df_merged = pd.merge(df_mh, df_wb_pivoted, left_on=['Code', 'Year'], right_on=['Country Code', 'Year'], how='inner')
df_merged = df_merged.drop(columns=['Country Code', 'Country Name'])

output_path = os.path.join(data_dir, 'merged_raw.csv')
df_merged.to_csv(output_path, index=False)
print(f"Dataset '{output_path}' gerado com sucesso!")