"""Método IEC 60599:2015 (Tabela-7 da dissertação).

Razões: C2H2/C2H4, CH4/H2, C2H4/C2H6. Método não possui diagnóstico "Normal".
"""

import pandas as pd

from dados import carregar_dataset

FAIXA_PARA_CLASSE3 = {
    "Descarga parcial": "Eletrica",
    "Baixa energia": "Eletrica",
    "Alta energia": "Eletrica",
    "Termica T<300C": "Termica",
    "Termica T<700C": "Termica",
    "Termica T>700C": "Termica",
}


def _razao(num: float, den: float) -> float:
    if den == 0:
        return float("inf") if num > 0 else 0.0
    return num / den


def diagnostico(h2: float, ch4: float, c2h4: float, c2h6: float, c2h2: float) -> str:
    r1 = _razao(c2h2, c2h4)  # C2H2/C2H4
    r2 = _razao(ch4, h2)  # CH4/H2
    r3 = _razao(c2h4, c2h6)  # C2H4/C2H6

    if r2 < 0.1 and r3 < 0.2:
        return "Descarga parcial"
    if r1 > 1.0 and 0.1 <= r2 <= 0.5 and r3 > 1.0:
        return "Baixa energia"
    if 0.6 <= r1 <= 2.5 and 0.1 <= r2 <= 1.0 and r3 > 2.0:
        return "Alta energia"
    if r2 > 1.0 and r3 < 1.0:
        return "Termica T<300C"
    if r1 < 0.1 and r2 > 1.0 and 1.0 <= r3 <= 4.0:
        return "Termica T<700C"
    if r1 < 0.2 and r2 > 1.0 and r3 > 4.0:
        return "Termica T>700C"
    return "Indeterminado"


def classificar(df: pd.DataFrame) -> pd.Series:
    diag = df.apply(
        lambda linha: diagnostico(
            linha["H2"], linha["CH4"], linha["C2H4"], linha["C2H6"], linha["C2H2"]
        ),
        axis=1,
    )
    return diag.map(FAIXA_PARA_CLASSE3).fillna("Indeterminado")


if __name__ == "__main__":
    from avaliacao import Resultado, imprimir_resultado

    df = carregar_dataset()
    pred = classificar(df)
    imprimir_resultado(Resultado("IEC 60599:2015", df["classe3"].tolist(), pred.tolist()))
