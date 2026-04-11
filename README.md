# estudo_saude_mental

## Datasets:

ihme_original.csv:
Dataset obtido pelo site do IHME (Institute for Health Metrics and Evaluation). O Dataset contém dados de diversos países, bem como alguns agregados (União Europeia, países de baixa renda, etc). Os dados que o Dataset contém são o PIB/GDP (NY.GDP.MKTP.CD), porcentagem da população com acesso a internet (IT.NET.USER.ZS), Gini Index/ índice de desigualdade social (SI.POV.GINI), porcentagem da população desempregada (SL.UEM.TOTL.NE.ZS), porcentagem da população desempregada por nível de escolaridade (SL.UEM.ADVN.ZS (etc)), porcentagem da população vivendo em área urbana (SP.URB.TOTL.IN.ZS), velocidade da urbanização anual (SP.URB.GROW) e total da dívida do governo central do pais (GC.DOD.TOTL.GD.ZS). O Dataset contém dados reais de 1993 até 2023, e alguns dados inferidos para 2024 e 2025.

kaggle_orginal.csv:
Datset obtido pelo Kaggle, com autoria de Mohamadreza Momeni. O Datset contém a porcentagem da população que apresenta Esquizofrenia, Depressão, Ansiedade, Transtorno Bipolar e Transtorno Alimentar em diversos países de 1990 até 2019.

## Análise Exploratória dos Dados:

Para a análise de ihme_original.csv, '..' (o símbolo usado pelo Datset para valores faltantes) foi alterado para NaN, linhas sem 'Country Code' foram removidas (para remover os agregados de países da análise e tratar apenas países individuais) e dados numéricos que haviam sido escritos em texto foram convertidos para valores numéricos. Foram gerados quatro gráficos para a análise:

eda_mh_disrtibutions.png:
Representa a distribuição das porcentagens de cada doença, considerando todos os países e todos os anos. O gráfico mostra que Depressão e Ansiedade são as doenças mais comuns, tendo suas curvas mais a direira, prevalência entre 3% a 5%. As demais doenças são menos comuns. Os gráficos da Depressão, da Ansiedade, da Esquizofrenia e da Bipolaridade se assemelham com uma distribuição normal. Contudo, o gráfico do Transtorno Alimentar forma uma cauda longa para a direita, destacando que em alguns países específicos os índices de Transtorno Alimentar são muito maiores que a média, que por si só é baixa.

eda_mh_correlation.png:
Representa a matriz de correlação entre as doençais mentais abordadas. A matriz apresenta as maiores correlações entre Bipolaridade, Ansiedade e Transtornos Alimentares, indicando que, nos casos em que um sobe, os outros tendem a subir junto. Surpreendentemente, a Depressão tem uma correlação baixa com a Ansiedade, sugerindo que as causas para Depressão podem varias das causas para Ansiedade. A correlação da Depressão com a Esquizofrenia foi negativa, o que não necessariamente indica que uma evita a existência da outra, mas sim que seus padrões de crescimento seguem direções opostas.

eda_outliers.png:
Representa a quantidade de valores que fogem do padrão, para cada doença. O gráfico mostra que a Esquizofrenia é a doença com valor mais concentrado, se mantendo um valor parecido para a maioria dos países. Já a Ansiedade, a Depressão e os Transtornos Alimentares apresentam muitos outliers, mostrando que existem bolhas de países com valores muito acima da média, com destaque para a Ansiedade, que tem casos com mais que o dobro da média (que são o Brasil entre 2004 a 2008).

eda_wb_distributions.png:
Representa a distribuição dos principais dados econômicos de todos os países no ano de 2015. A tabela ofertada pelo IHME contém anos diferentes em colunas diferentes, portanto, para simplificar a análise inicial apenas o ano de 2015 foi avaliado. Na avaliação do PIB/ GDP, vemos um pico enorme em valores baixissímos e uma linha que quase desaparece conforme crescem os valores. Isso demonstra que existe um número enorme de países com PIB menor que 1 trilhão de dólares, porém alguns poucos países que apresentam valores até 5 vezes maior, expremendo a maioria dos países no lado esquerdo do gráfico dificultando sua análise. Isso mostra que o PIB será um dado que precisará de tratamento. A distribuição do desemprego tem uma leve cauda para a direita, como esperado. A maioria dos países possui um índice de desemrpego entre 0% e 10%, com alguns países ultrapassando 25%, que já representa um país com a economia em crise. Assim como o desemprego, o índice Gini apresenta a distribuição esperada, com uma concetração de países com valores menores (representam menor desigualdade social) e poucos países com valores maiores (representam maior desigualdade social), mas no geral bem distribuídos.