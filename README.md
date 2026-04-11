# estudo_saude_mental

## Datasets:

ihme_original.csv:
Dataset obtido pelo site do IHME (Institute for Health Metrics and Evaluation). O Dataset contém dados de diversos países, bem como alguns agregados (União Europeia, países de baixa renda, etc). Os dados que o Dataset contém são o PIB/GDP (NY.GDP.MKTP.CD), porcentagem da população com acesso a internet (IT.NET.USER.ZS), Gini Index/ índice de desigualdade social (SI.POV.GINI), porcentagem da população desempregada (SL.UEM.TOTL.NE.ZS), porcentagem da população desempregada por nível de escolaridade (SL.UEM.ADVN.ZS (etc)), porcentagem da população vivendo em área urbana (SP.URB.TOTL.IN.ZS), velocidade da urbanização anual (SP.URB.GROW) e total da dívida do governo central do pais (GC.DOD.TOTL.GD.ZS). O Dataset contém dados reais de 1993 até 2023, e alguns dados inferidos para 2024 e 2025.

kaggle_orginal.csv:
Datset obtido pelo Kaggle, com autoria de Mohamadreza Momeni. O Datset contém a porcentagem da população que apresenta Esquizofrenia, Depressão, Ansiedade, Transtorno Bipolar e Transtorno Alimentar em diversos países de 1990 até 2019.

## Análise Exploratória dos Dados:

Para a análise de ihme_original.csv, '..' (o símbolo usado pelo Datset para valores faltantes) foi alterado para NaN, linhas sem 'Country Code' foram removidas (para remover os agregados de países da análise e tratar apenas países individuais) e dados numéricos que haviam sido escritos em texto foram convertidos para valores numéricos.