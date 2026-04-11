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

grafico01_cobertura_geral.png:
Apresenta a cobertura de cada variável como percentual de entradas preenchidas sobre o total de países e anos. As variáveis do dataset do Kaggle (GBD) têm cobertura de 100% para todos os transtornos, por que o próprio GBD preenche países sem dados com estimativas feitas com uma metaregressão bayesiana, ou seja, muitas vezes representam uma estimativa do IHME. Já as váriaveis do World Bank, algumas apresentam uma cobertura não satisfatória, o que torna necessário a aplicação de técnicas de manipulação para dados faltantes.

grafico02_cobertura_anual_wb.png:
Detalha a cobertura das variáveis no dataset do World Bank, representando como a distribuição se dá ao longo do tempo. Esse gráfico ajuda a visualização para que seja verificada uma necessidade de tratamento de dados faltantes baseados nos anos, que não será necessário.

grafico03_gaps_(variavel).png:
Apresentam uma melhor visualização da distribuição dos gaps por país com o uso de um histograma, assim como um gráfico sobre a cobertura de cada variável ao longo dos anos. Os gráficos revelam a necessidade de descarte da variável de dívida pública, visto que possui um gap muito grande para muitos países, e existem poucos países em que esse gap é menor que 5. Além disso, a partir da observação dos gráficos, é possível notar a necessidade de técnicas de preenchimento de dados faltantes em gaps menores, com métodos de mineração de dados ou de interpolação.

grafico04_gdp_assimetria.png:
Comparação entre a distribuição bruta do PIB numa snapshot de 2010 com o resultado da transformação log. Na distribuição bruta praticamente todos os países ficam próximos de 0, com uma cauda que se extende até 15.000 bilhões de USD, com um skewness de 8.56. Após a tranformação, a distribuição passa a ser mais simétrica, com um skewness de -0.01.

grafico05_gdp_skew_temporal.png:
Confirma a prevalência da assimetria observada no ano de 2010 para todos os anos analisados.

grafico06_distribuicoes_wb.png:
Apresenta a distribuição das variáveis Desemprego Total, Gini e Internet, assim como o skewness de cada um. Inicialmente não será feita nenhuma técnica para redução da assimetria visualizada.

grafico07_distribuicoes_kaggle.png:
Apresenta a distribuição dos transtornos mentais no dataset do Kaggle, assim como o skewness de cada um. Inicialmente não será feita nenhuma técnica para redução da assimetria visualizada.

grafico08_outliers_gdp.png:
Gráficos boxplot para visualizar a presença dos outliers no PIB total por ano, reforçando a necessidade da transformação log ou técnicas para a redução da assimetria presente na distribuição dos valores dessa variável.

## Pré-processamento:

**importante explicar o melt

O primeiro passo do pré-processamento foi a tranformação da tabela de ihme_original.csv. Como citado, esse dataset apresenta cada ano em uma coluna diferente (cada ano um atributo de uma instância), enquanto o dataset do kaggle apresenta cada ano em uma linha (cada ano como uma instância isolada). Para corrigir esse problema, foi feito o melt da tabela usando pd.melt. Além disso, as colunas de anos do dataset estavam escritas como '2000 [YR2000]', essas colunas (que se tornaram linhas) foram renomeadas para apenas o número do ano usando expressões regulares. Além disso, o rodapé foi removido e os dados vazios (formatados pelo banco como '..') foram formatados para np.nan. O dataset após o melt foi salvo em ihme_melted.csv.

Após o melt, os indicadores da tabela original (PIB, GiniIndex, etc) ficaram todos em uma única coluna, cada um definindo uma instância junto do ano (como era no datset original). Porém, para nosso estudo, cada indicador desse precisa estar em uma coluna diferente, sendo cada instância definida por país e ano, apenas. Para isso, foi usado o método pivot_table da biblioteca pandas. A nova tabela já pivotada foi salva em ihme_pivoted.csv.

Agora as duas tabelas podem ser unidas. Ambas possuem como primeiras colunas: nome do país, código do país e ano, seguidos das colunas com as informações de cada tabela específica. Porem, antes do merge, os nomes das colunas com índices das doenças em kaggle_original.csv foram renomeadas de 'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized' para 'Preval_Esquizofrenia', por exemplo.

Para o merge das duas tabelas foi usado um inner join. Essa escolha é estratégica pois, graças ao inner join, apenas as instâncias que tiverem o mesmo código de país e o mesmo ano serão transcritas para a tabela final, removendo automaticamente instâncias que só estão presentes em uma tabela. Isso irá descartar automaticamente os anos de 1990-1993 que só existem na tabela do Kaggle e os anos de 2020-2025 que só existem na tabela do IHME, além dos agrupamentos de países citados que só existem na tabela do IHME. A chave para a junção será o código do país e o ano, e os dados são salvos em merged_raw.csv.

Após o merge ainda existem alguns problemas, como, por exemplo, muitos dados faltantes, dados com muita discrepância, entre outros. Todos esses problemas podem impactar significamente na qualidade do modelo e do conhecimento gerado futuramente. Portanto, a técnica escolhida para lidar com cada um desses problemas deve ser escolhida cautelosamente. 