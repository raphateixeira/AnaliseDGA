# AnaliseDGA

Sistema de Diagnóstico Inteligente de Transformadores de Potência por
Análise DGA (Análise de Gases Dissolvidos em óleo), com os métodos ANN
(Redes Neurais Artificiais), KNN (k-vizinhos mais próximos) e DUVAL, e o
conjunto de dados usado para treinar e avaliar os modelos.

Trabalho da dissertação de mestrado intitulada _Diagnóstico inteligente de
faltas em transformadores baseado na análise de gás dissolvido em óleo_
([ver publicação](https://repositorio.ufpa.br/items/eeecc00f-702c-453c-968a-867779a14a70),
PDF incluído neste repositório em `Publicacoes/Dissertacao_Otacilio.pdf`).

- **Autor:** Otacílio Rodrigues Oliveira Filho
- **Orientador:** Prof. Dr. Raphael Barros Teixeira

🔗 Site: https://raphateixeira.github.io/AnaliseDGA/

## Conteúdo

- `Apresentacao/apresentacao.qmd` — apresentação (Quarto Reveal) comparando os resultados do mestrado com os novos resultados.
- `ResultadosMestrado/` — código e modelos originais da dissertação:
  - `SistemaDiagnosticoInteligente.ipynb`, `Knn.ipynb`, `RNA.ipynb` — notebooks do sistema de diagnóstico (ANN, KNN, DUVAL).
  - `knn_DGA.sav`, `meuRnn.pkl` — modelos treinados serializados.
- `ResultadosNovos/` — novos modelos, em desenvolvimento.
- `DataSet/` — dados brutos:
  - `DataSetDGA.xlsx` — conjunto de dados de gases dissolvidos em óleo (2004 amostras), migrado do antigo repositório `Dataset-DGA`.
- `Publicacoes/` — documentos publicados/a publicar:
  - `Dissertacao_Otacilio.pdf` — texto completo da dissertação (repositório da UFPA, acesso aberto).
  - `Artigo_PPCA_DGA.pdf` — artigo derivado da dissertação, a ser submetido.

Este repositório é o antigo `SDI-DGA` (renomeado), com o conteúdo do antigo
`Dataset-DGA` incorporado. Ambos tratavam do mesmo trabalho.
