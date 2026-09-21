# ResultadosNovos

Benchmark próprio dos métodos de diagnóstico DGA estudados na
[dissertação](../Publicacoes/Dissertacao_Otacilio.pdf) de Otacílio Rodrigues
Oliveira Filho (UFPA, 2024), sobre o mesmo [DataSet](../DataSet/), sob um
protocolo de avaliação único e reprodutível — corrigindo por construção uma
série de problemas metodológicos identificados na primeira tentativa
(`rascunho_v1/`, mantida só como referência histórica).

## Como rodar

```bash
python3 ResultadosNovos/benchmark_dga.py                    # benchmark completo
python3 ResultadosNovos/benchmark_dga.py --ablacao zeros       # ablação: tratamento do piso de 0,01 ppm
python3 ResultadosNovos/benchmark_dga.py --ablacao outliers    # ablação: outliers K-means
python3 ResultadosNovos/benchmark_dga.py --ablacao representacao  # ablação: atributos básicos x estendidos
```

Stack: `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib` (sem TensorFlow/Keras — ver
limitação da ANN abaixo). Saída completa em `resultados/` (gerada pelo script, não versionada
manualmente): `tabelas/*.csv`, `tabelas/tabela_I.tex`, `tabelas/tabela_II.tex`,
`figuras/*.pdf`, `predicoes/*.csv` (vetores pareados por modelo).

## O que a primeira tentativa (`rascunho_v1/`) errou

1. Cada modelo era avaliado em um conjunto de teste diferente — comparação e Kappa entre
   modelos inválidos.
2. Métodos convencionais (Duval, Rogers etc.) eram avaliados só nas amostras de falta e só
   nas com diagnóstico determinado — comparação injusta contra os métodos de AM, que eram
   avaliados no dataset inteiro.
3. *Hold-out* único, sem dispersão nem teste de hipótese.
4. Desbalanceamento (65% normais) declarado e não tratado; acurácia bruta como métrica
   principal.

Este script torna cada um desses erros impossível por construção: **um único laço de
avaliação** (`avaliar_todos`) roda `cross_val_predict` com a **mesma** partição de
`StratifiedGroupKFold` (10 folds × 5 repetições) para todos os 12 modelos, com uma
`assert` explícita garantindo que toda predição cobre as 2004 amostras (nunca um
subconjunto) — o teste de regressão contra o erro nº1. Os convencionais são avaliados no
dataset inteiro, incluindo amostras normais mesmo quando o método não tem diagnóstico
"normal" (`suporta_normal=False` em Duval e IEC 60599) — o erro nº2 vira, ao contrário,
o próprio resultado mais importante do benchmark (ver abaixo).

## Achados da auditoria de dados (antes de treinar qualquer coisa)

- **2004 amostras, 3 classes agrupadas** (mesmo mapeamento da dissertação e da coluna
  `NTE` do dataset): `normal` 64,6% (1295) / `T` 22,1% (442) / `D` 13,3% (267).
- **Não há zero real no dataset.** `min()` de todo gás é exatamente 0,01 ppm — o piso já
  vem aplicado no arquivo-fonte (73,5% das linhas têm ≥1 gás em 0,01; C2H2 sozinho: 71,9%).
  Não há como recuperar os zeros originais. `TrataZeros` trata `valor==0.01` como
  sentinela de "não detectado", não `valor==0`.
- **264 linhas (13,2%) têm composição de gases idêntica a outra linha** — a maioria (260)
  rotuladas `normal`. Um par tem **rótulos conflitantes**: mesma composição
  (H2=6, CH4=3, C2H4=C2H6=C2H2=0,01) aparece como `NO` numa linha e `D1` noutra — ruído de
  rótulo genuíno, mantido no fluxo principal (nunca removemos amostras), mas usado como
  `groups` no `StratifiedGroupKFold` para que cópias idênticas nunca caiam uma em treino e
  outra em teste no mesmo fold.
- **`NTE` e `NF` não são "códigos de norma/falta"** (como o `Dataset.qmd` os descrevia) —
  são o próprio rótulo codificado em número (`NTE`: 0/1/2 = normal/T/D; `NF`: 0/1 =
  normal/falha), correspondência perfeita. Ficam fora de qualquer pipeline.

## Resultado principal — desempenho global (10 folds × 5 repetições)

| Método | Acurácia | Acc. balanceada | F1-macro | Taxa indeterminação |
|---|---|---|---|---|
| Duval | 28,3% | 49,8% | 29,0% | 5,2% |
| Rogers | 23,6% | 39,6% | 33,4% | 42,3% |
| Gás Chave | 53,6% | 62,4% | 51,8% | 0,0% |
| Doernenburg | 70,9% | 54,6% | 66,0% | 22,6% |
| IEC 60599 | 25,4% | 43,0% | 35,0% | 41,5% |
| LR | 89,3% | 90,9% | 87,4% | 0,0% |
| SVM (10 PCs + atributos) | 91,9% | 92,3% | 90,1% | 0,0% |
| SVM (5 gases) | 91,9% | 91,9% | 90,0% | 0,0% |
| KNN | 94,6% | 91,8% | 92,7% | 0,0% |
| KNN (atributos estendidos) | 94,0% | 90,8% | 92,0% | 0,0% |
| ANN (MLPClassifier) | 93,2% | 89,9% | 91,2% | 0,0% |
| **HistGB** | **95,3%** | **93,1%** | **93,7%** | 0,0% |

Tabelas completas (todas as métricas, desvio-padrão entre repetições, precisão/revocação
por classe): `resultados/tabelas/metricas_globais.csv`, `tabela_I.tex`, `tabela_II.tex`.

### Os convencionais desabam quando avaliados de forma justa

Na dissertação original, os métodos convencionais eram avaliados só nas amostras de
falha com diagnóstico determinado. Aqui, avaliados no dataset inteiro (incluindo os 65%
de amostras normais, nas quais Duval e IEC 60599 **nunca** acertam por não terem essa
classe), a acurácia deles cai para 24–71%. Isso não é uma falha da implementação — é
exatamente o vazamento metodológico que motivou este benchmark. O Gás Chave e o
Doernenburg (que suportam diagnóstico normal) se saem relativamente melhor (54% e 71%)
justamente por isso.

### HistGB "vence", mas não com significância estatística sobre o KNN

`HistGradientBoostingClassifier` (não avaliado na dissertação original) obteve a maior
acurácia pontual (95,3%), superando o KNN (94,6%, campeão do trabalho original). **Reportando sem
atenuar, como pedido:** o teste de McNemar pareado (repetição 0, correção de Holm) entre
HistGB e KNN dá **p = 0,54** — a diferença **não é estatisticamente significativa**. Ou
seja, o "novo campeão" é um empate técnico com o KNN, não uma vitória clara.
Ver `resultados/tabelas/mcnemar_holm.csv` para todos os 66 pares.

### KNN e ANN — o par escolhido para a arquitetura híbrida original — são os mais concordantes

Kappa de Cohen (ponderação linear, repetição 0) entre KNN e ANN: **0,87** — o par de maior
concordância entre os modelos de AM (`resultados/tabelas/kappa_pares.csv`). Como o texto
da dissertação também observa concordância alta entre esses dois métodos, vale registrar:
**alta concordância indica redundância, não necessariamente qualidade** — dois
classificadores que quase sempre concordam agregam pouca informação nova um ao outro
quando combinados em cascata (a arquitetura SDI original combina exatamente KNN + ANN +
Duval).

### Ablação — representação básica vs. estendida: sem ganho claro

Isolando o efeito da representação de entrada do efeito do classificador (mesmo
classificador, dois pré-processadores):

| Par | Básico (5 gases) | Estendido (+razões +%Duval) |
|---|---|---|
| KNN | **94,6%** | 94,0% |
| SVM | 91,9% | **91,9%** (empate) |

Os atributos engenheirados (razões de Rogers, percentuais de Duval) **não melhoraram** o
desempenho de forma consistente — no KNN até pioraram levemente. Isso contrasta com a
premissa da dissertação original, onde só o SVM recebia atributos engenheirados (tornando
impossível separar o efeito do classificador do efeito da representação); aqui, com o
mesmo classificador nas duas condições, o ganho não se sustenta.

### Ablação — tratamento do piso de 0,01 ppm (classificador LR)

| Estratégia | Acurácia | Acc. balanceada | F1-macro |
|---|---|---|---|
| `log1p` (padrão) | 89,0% | 90,6% | 87,1% |
| `lod_metade` (metade do LOD por gás) | 89,1% | 90,6% | 87,6% |
| `original` (0,01 + log, sem log1p) | 87,6% | 89,6% | 85,3% |

Tratar o piso de forma ingênua (`original`, reproduzindo o artefato do trabalho anterior)
mede-se consistentemente pior nas 3 métricas — pequeno mas real.

### Ablação — outliers por K-means entre amostras normais

Não temos a receita exata do K-means da dissertação (removeu 181/1295 = 14,0% das
amostras normais). Testamos primeiro "cluster minoritário de um k=2": marcou 43,5% das
normais como outlier — implausível (é bisseção da classe, não detecção de anomalia).
Trocamos para o critério padrão de **distância ao centroide** (K-means com k=4, top 10%
mais distantes do próprio centroide):

| Versão | Amostras | Acurácia | Acc. balanceada | F1-macro |
|---|---|---|---|---|
| Com outliers (dataset completo) | 2004 | 89,0% | 90,6% | 87,1% |
| Sem outliers (130 normais removidas, 10,0%) | 1874 | 89,9% | 91,2% | 88,7% |

Remover as amostras normais mais distantes do seu centroide infla a acurácia em
~0,8–1,3 ponto percentual — real, mas bem mais modesto que a diferença de 5+ pontos que
apareceria com o critério de bisseção k=2. Fica registrado como limitação: não sabemos se
o critério original da dissertação era mais parecido com um ou outro.

## Limitações documentadas

- **ANN**: `MLPClassifier` do sklearn não tem ativação `selu` nem `class_weight` (a rede
  original da dissertação usava Keras com arquitetura 12-12-8-6, `selu`, `adam` com
  learning rate/decay/clipvalue customizados). Aqui: `relu`, `adam` padrão,
  `early_stopping=True`. O desbalanceamento é compensado via `class_weight="balanced"`
  nos demais modelos, mas não na ANN.
- **Sintonia de hiperparâmetros com CV interna sem agrupamento**: a busca
  (`RandomizedSearchCV`) usa `StratifiedKFold` simples na CV interna, não
  `StratifiedGroupKFold`. Uma duplicata pode cair em treino-interno e validação-interna da
  mesma busca, enviesando levemente a escolha de hiperparâmetros — mas isso nunca vaza
  para a avaliação externa reportada (o `StratifiedGroupKFold` externo garante que cada
  grupo de composição idêntica fica inteiro em um único fold de teste). Implementar
  agrupamento também na busca interna exigiria roteamento de metadados do sklearn,
  descartado aqui por complexidade frente ao ganho.
- **Piso de 0,01 ppm**: não é possível recuperar os zeros originais (ver auditoria acima);
  a estratégia `"original"` da ablação de zeros reproduz o artefato sobre o dado já
  floored, não sobre zero real.
- **Gás Chave e Doernenburg**: o dataset não tem a coluna CO — implementados só com os 5
  gases disponíveis (mesma limitação que a própria dissertação assume para Doernenburg).
- **Duval Triangle 1**: geometria das 7 regiões reconstruída visualmente a partir da
  Figura-8 do PDF da dissertação e validada contra os 2 casos canônicos da Tabela-2 (a
  amostra de falha T3 cai exatamente na região T3) — ver `benchmark_dga.py` para os
  vértices exatos.

## Estrutura

- `benchmark_dga.py` — script único com tudo (transformadores, 5 métodos convencionais
  como estimadores sklearn, dicionário `MODELOS`, sintonia, laço de avaliação, testes
  estatísticos, exportação, ablações).
- `resultados/` — gerado pelo script (`tabelas/`, `figuras/`, `predicoes/`).
- `rascunho_v1/` — primeira tentativa (mantida como referência histórica; não usar).
