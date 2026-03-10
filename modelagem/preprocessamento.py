"""
preprocessamento.py
===================
Limpeza, normalização e transformação dos dados brutos.
Projeto de Parceria Semantix — Evasão Escolar no Brasil
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("../coleta_de_dados/dados_brutos")
OUT_DIR = Path("dados_processados")
OUT_DIR.mkdir(exist_ok=True)


def limpar_censo_escolar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza e padronização do dataset do Censo Escolar.
    """
    log.info("🧹 Limpando dados do Censo Escolar...")
    df = df.copy()

    # Remover duplicatas
    antes = len(df)
    df = df.drop_duplicates(subset=["id_escola", "ano"])
    log.info(f"   Duplicatas removidas: {antes - len(df)}")

    # Tratar outliers na taxa de abandono (>35% considerado outlier)
    mask_outlier = df["taxa_abandono_pct"] > 35
    df.loc[mask_outlier, "taxa_abandono_pct"] = df.loc[~mask_outlier, "taxa_abandono_pct"].mean()
    log.info(f"   Outliers corrigidos: {mask_outlier.sum()}")

    # Padronizar categorias
    df["regiao"] = df["regiao"].str.strip().str.title()
    df["localizacao"] = df["localizacao"].str.strip().str.title()
    df["nivel_ensino"] = df["nivel_ensino"].str.strip()

    # Criar variável categórica de risco
    df["nivel_risco"] = pd.cut(
        df["taxa_abandono_pct"],
        bins=[0, 3, 8, 15, 100],
        labels=["Baixo", "Moderado", "Alto", "Crítico"],
        right=True
    )

    # Normalizar infraestrutura (0-100)
    df["infraestrutura_score_100"] = (df["infraestrutura_score"] * 100).round(0).astype(int)

    log.info(f"   ✅ Censo Escolar processado: {len(df)} registros")
    return df


def limpar_pnad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza e feature engineering da PNAD Contínua.
    """
    log.info("🧹 Limpando dados da PNAD Contínua...")
    df = df.copy()

    # Preencher nulos em motivo_evasao
    df["motivo_evasao"] = df["motivo_evasao"].fillna("Não evadiu")

    # Criar faixas etárias
    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[9, 12, 14, 16, 18, 25],
        labels=["10-12", "13-14", "15-16", "17-18", "19-24"]
    )

    # Criar faixas de renda
    df["faixa_renda"] = pd.cut(
        df["renda_per_capita"],
        bins=[0, 200, 500, 1000, 2000, float("inf")],
        labels=["Extrema Pobreza", "Pobreza", "Baixa Renda", "Média Renda", "Alta Renda"]
    )

    # Score de vulnerabilidade
    df["score_vulnerabilidade"] = (
        (df["renda_per_capita"] < 500).astype(int) * 2 +
        df["trabalha"].astype(int) * 3 +
        (df["anos_estudo_mae"] < 4).astype(int) * 1 +
        (1 - df["bolsa_familia"]).astype(int) * 1
    )

    log.info(f"   ✅ PNAD processada: {len(df)} registros")
    return df


def limpar_municipios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza e enriquecimento dos dados municipais.
    """
    log.info("🧹 Limpando dados municipais...")
    df = df.copy()

    # Classificação IDH
    df["classe_idh"] = pd.cut(
        df["idh"],
        bins=[0, 0.55, 0.70, 0.80, 1.0],
        labels=["Muito Baixo", "Baixo", "Médio", "Alto"]
    )

    # Quintis de abandono escolar
    df["quintil_abandono"] = pd.qcut(
        df["taxa_abandono_escolar"],
        q=5, labels=["Q1 (Menor)", "Q2", "Q3", "Q4", "Q5 (Maior)"]
    )

    # Log da taxa de homicídios (reduz skewness)
    df["log_homicidios"] = np.log1p(df["taxa_homicidios_100k"])

    log.info(f"   ✅ Municípios processados: {len(df)} registros")
    return df


def criar_dataset_consolidado(df_censo, df_pnad, df_mun) -> pd.DataFrame:
    """
    Cria dataset agregado por região para análises cruzadas.
    """
    log.info("🔗 Consolidando datasets por região...")

    # Agregar censo por região
    censo_reg = df_censo.groupby("regiao").agg(
        taxa_abandono_media=("taxa_abandono_pct", "mean"),
        n_escolas=("id_escola", "count"),
        pct_rural=("localizacao", lambda x: (x == "Rural").mean() * 100),
        infra_media=("infraestrutura_score", "mean")
    ).reset_index()

    # Agregar municípios por região
    mun_reg = df_mun.groupby("regiao").agg(
        idh_medio=("idh", "mean"),
        homicidios_medio=("taxa_homicidios_100k", "mean"),
        trabalho_infantil_medio=("pct_trabalho_infantil", "mean"),
        bolsa_familia_medio=("pct_beneficiarios_bolsa", "mean")
    ).reset_index()

    # PNAD: taxa de evasão por região (simulado por renda como proxy)
    df_consolidado = censo_reg.merge(mun_reg, on="regiao", how="left")
    df_consolidado = df_consolidado.round(3)

    log.info(f"   ✅ Dataset consolidado: {df_consolidado.shape}")
    return df_consolidado


def main():
    log.info("=" * 60)
    log.info("PRÉ-PROCESSAMENTO DE DADOS — EVASÃO ESCOLAR")
    log.info("=" * 60)

    # Tentar carregar ou gerar dados
    try:
        df_censo_raw = pd.read_csv(DATA_DIR / "censo_escolar_2023.csv")
        df_pnad_raw = pd.read_csv(DATA_DIR / "pnad_continua_2023.csv")
        df_mun_raw = pd.read_csv(DATA_DIR / "dados_municipios.csv")
    except FileNotFoundError:
        import sys
        sys.path.insert(0, str(Path("../coleta_de_dados")))
        from coleta_dados import gerar_dados_censo_escolar, gerar_dados_pnad, gerar_dados_municipios
        df_censo_raw = gerar_dados_censo_escolar()
        df_pnad_raw = gerar_dados_pnad()
        df_mun_raw = gerar_dados_municipios()

    # Processar
    df_censo = limpar_censo_escolar(df_censo_raw)
    df_pnad = limpar_pnad(df_pnad_raw)
    df_mun = limpar_municipios(df_mun_raw)
    df_consolidado = criar_dataset_consolidado(df_censo, df_pnad, df_mun)

    # Salvar
    df_censo.to_csv(OUT_DIR / "censo_processado.csv", index=False)
    df_pnad.to_csv(OUT_DIR / "pnad_processada.csv", index=False)
    df_mun.to_csv(OUT_DIR / "municipios_processados.csv", index=False)
    df_consolidado.to_csv(OUT_DIR / "consolidado_regional.csv", index=False)

    log.info("\n✅ Pré-processamento concluído!")
    log.info(f"📁 Arquivos salvos em: {OUT_DIR.resolve()}")

    # Relatório de qualidade pós-processamento
    print("\n" + "="*50)
    print("RELATÓRIO PÓS-PROCESSAMENTO")
    print("="*50)
    print(f"Censo Escolar    : {len(df_censo):>6} registros | {df_censo['nivel_risco'].value_counts().to_dict()}")
    print(f"PNAD Contínua    : {len(df_pnad):>6} registros | Evasão: {df_pnad['evadiu_escola'].mean():.1%}")
    print(f"Dados Municipais : {len(df_mun):>6} municípios | IDH médio: {df_mun['idh'].mean():.3f}")
    print(f"Consolidado      : {len(df_consolidado):>6} regiões")

    return df_censo, df_pnad, df_mun, df_consolidado


if __name__ == "__main__":
    main()
