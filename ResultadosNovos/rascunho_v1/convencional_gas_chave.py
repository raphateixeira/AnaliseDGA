"""Método do Gás Chave (Tabela-4 da dissertação).

O dataset não possui a coluna CO, portanto o perfil "Monóxido de Carbono /
Térmico papel-óleo" não pode ser avaliado (mesma limitação do Doernenburg
neste repositório). A classificação usa o gás de maior concentração relativa
dentre os três disponíveis (C2H4, H2, C2H2) como "gás chave".
"""

import pandas as pd

from dados import carregar_dataset

GAS_CHAVE_PARA_CLASSE3 = {
    "C2H4": "Termica",  # Etileno -> Térmica-óleo
    "H2": "Eletrica",  # Hidrogênio -> Baixa energia
    "C2H2": "Eletrica",  # Acetileno -> Alta energia
}


def diagnostico(h2: float, ch4: float, c2h4: float, c2h6: float, c2h2: float) -> str:
    valores = {"H2": h2, "CH4": ch4, "C2H4": c2h4, "C2H6": c2h6, "C2H2": c2h2}
    total = sum(valores.values())
    if total == 0:
        return "Indeterminado"
    # Se o gás dominante não é um "gás chave" da Tabela-4 (CH4 ou C2H6),
    # nenhuma falha é apontada pelo método -> Normal.
    dominante = max(valores, key=valores.get)
    if dominante not in GAS_CHAVE_PARA_CLASSE3:
        return "Normal"
    return GAS_CHAVE_PARA_CLASSE3[dominante]


def classificar(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda linha: diagnostico(
            linha["H2"], linha["CH4"], linha["C2H4"], linha["C2H6"], linha["C2H2"]
        ),
        axis=1,
    )


if __name__ == "__main__":
    from avaliacao import Resultado, imprimir_resultado

    df = carregar_dataset()
    pred = classificar(df)
    imprimir_resultado(Resultado("Gas chave", df["classe3"].tolist(), pred.tolist()))

