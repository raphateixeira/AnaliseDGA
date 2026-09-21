"""Carregamento do DataSet DGA e agrupamento das 7 classes em 3 (Normal/Térmica/Elétrica).

O agrupamento em 3 classes segue a metodologia da dissertação (Capítulo 6):
NO -> Normal; T1/T2/T3 -> Térmica; D1/D2/PD -> Elétrica.
"""

from pathlib import Path

import pandas as pd

CAMINHO_DATASET = Path(__file__).resolve().parent.parent / "DataSet" / "DataSetDGA.xlsx"

GASES = ["H2", "CH4", "C2H4", "C2H6", "C2H2"]

GRUPO_3_CLASSES = {
    "NO": "Normal",
    "T1": "Termica",
    "T2": "Termica",
    "T3": "Termica",
    "D1": "Eletrica",
    "D2": "Eletrica",
    "PD": "Eletrica",
}

ORDEM_3_CLASSES = ["Normal", "Termica", "Eletrica"]


def carregar_dataset(caminho: Path = CAMINHO_DATASET) -> pd.DataFrame:
    df = pd.read_excel(caminho)
    df = df.rename(columns={"Rótulo": "rotulo"})
    df["classe3"] = df["rotulo"].map(GRUPO_3_CLASSES)
    return df[GASES + ["rotulo", "classe3"]]
