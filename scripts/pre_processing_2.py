import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'datasets')

df_merged = pd.read_csv(os.path.join(data_dir, 'merged_raw.csv'))

# # --- 5. Imputação (Tratamento de Dados Faltantes) ---
# # Usamos Interpolação Linear agrupada por país. 
# # Se um país tem dados em 2010 e 2012, ele estima 2011.
# cols_numeric = df_merged.select_dtypes(include=[np.number]).columns.difference(['Year'])
# df_merged[cols_numeric] = df_merged.groupby('Code')[cols_numeric].transform(
#     lambda x: x.interpolate(limit_direction='both')
# )

# # Removemos linhas que ainda restaram nulas (países sem NENHUMA informação do indicador)
# df_final = df_merged.dropna()

# # --- 6. Engenharia de Atributos e Escalonamento ---
# # Aplica Log no PIB para reduzir a disparidade (que vimos na EDA)
# if 'GDP (current US$)' in df_final.columns:
#     df_final['GDP_Log'] = np.log1p(df_final['GDP (current US$)'])
#     df_final = df_final.drop(columns=['GDP (current US$)'])

# # Escalonamento Z-Score (Média 0, Desvio Padrão 1)
# # Isso coloca Ansiedade, Gini e PIB na mesma "importância" para a IA
# scaler = StandardScaler()
# cols_to_scale = df_final.select_dtypes(include=[np.number]).columns.difference(['Year'])
# df_final_scaled = df_final.copy()
# df_final_scaled[cols_to_scale] = scaler.fit_transform(df_final[cols_to_scale])

# # --- 7. Salvamento ---
# output_path = os.path.join(data_dir, 'dataset_final_processado.csv')
# df_final_scaled.to_csv(output_path, index=False)
# print(f"Dataset '{output_path}' gerado com sucesso!")