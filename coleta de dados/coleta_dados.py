"""
coleta_dados.py
===============
Script de coleta automatizada de dados públicos sobre evasão escolar no Brasil.
Projeto de Parceria Semantix — Análise de Dados Educacionais

Fontes:
- INEP (Censo Escolar)
- IBGE (PNAD Contínua)
- MDS (Bolsa Família / CadÚnico)
- PNUD (IDH Municipal)
- IPEA (Atlas da Violência)

Requisitos:
    pip install requests pandas numpy tqdm openpyxl
"""

import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("coleta.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dados_brutos"
DATA_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Simulação de dados (ambiente de desenvolvimento)
# Em produção, substituir pelos downloads reais
# ──────────────────────────────────────────────

def gerar_dados_censo_escolar(n_escolas: int = 5000) -> pd.DataFrame:
    """
    Simula microdados do Censo Escolar INEP.
    Em produção: baixar de https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar
    """
    np.random.seed(42)
    regioes = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    pesos_regiao = [0.15, 0.30, 0.10, 0.30, 0.15]
    estados = {
        "Norte": ["AM", "PA", "TO", "RO", "AC", "RR", "AP"],
        "Nordeste": ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"],
        "Centro-Oeste": ["MT", "MS", "GO", "DF"],
        "Sudeste": ["SP", "RJ", "MG", "ES"],
        "Sul": ["PR", "SC", "RS"]
    }
    niveis = ["Fundamental I", "Fundamental II", "Ensino Médio"]
    localizacoes = ["Urbana", "Rural"]

    dados = []
    for i in range(n_escolas):
        regiao = np.random.choice(regioes, p=pesos_regiao)
        uf = np.random.choice(estados[regiao])
        nivel = np.random.choice(niveis, p=[0.35, 0.35, 0.30])
        local = np.random.choice(localizacoes, p=[0.75, 0.25])

        # Taxa de abandono varia por região e nível
        base_abandono = {
            "Norte": 12.5, "Nordeste": 11.2,
            "Centro-Oeste": 7.8, "Sudeste": 5.1, "Sul": 3.9
        }[regiao]

        nivel_mult = {"Fundamental I": 0.6, "Fundamental II": 1.0, "Ensino Médio": 1.8}[nivel]
        local_mult = 1.5 if local == "Rural" else 1.0
        ruido = np.random.normal(0, 1.5)

        taxa_abandono = max(0, min(40, base_abandono * nivel_mult * local_mult + ruido))

        dados.append({
            "id_escola": f"ESC{i+1:05d}",
            "ano": np.random.choice([2021, 2022, 2023]),
            "regiao": regiao,
            "uf": uf,
            "localizacao": local,
            "nivel_ensino": nivel,
            "taxa_abandono_pct": round(taxa_abandono, 2),
            "total_matriculas": np.random.randint(50, 1200),
            "infraestrutura_score": round(np.random.uniform(0.2, 1.0), 2),
            "tem_internet": np.random.choice([0, 1], p=[0.25, 0.75]),
            "tem_biblioteca": np.random.choice([0, 1], p=[0.40, 0.60]),
            "distancia_km": round(np.random.exponential(3), 1) if local == "Rural" else 0.0,
        })

    return pd.DataFrame(dados)


def gerar_dados_pnad(n_individuos: int = 10000) -> pd.DataFrame:
    """
    Simula microdados da PNAD Contínua (IBGE).
    Em produção: baixar de https://ftp.ibge.gov.br/...
    """
    np.random.seed(123)
    dados = []

    for i in range(n_individuos):
        idade = np.random.randint(10, 25)
        renda = max(0, np.random.lognormal(6.5, 0.8))  # Renda per capita
        sexo = np.random.choice(["M", "F"])
        raca = np.random.choice(["Branca", "Preta", "Parda", "Amarela", "Indígena"],
                                 p=[0.43, 0.09, 0.47, 0.005, 0.005])
        trabalha = 1 if (renda < 800 and np.random.random() < 0.35) else 0
        bolsa_familia = 1 if renda < 500 else 0

        # Calcular probabilidade de evasão
        prob_evasao = 0.05
        if renda < 500: prob_evasao += 0.15
        if trabalha: prob_evasao += 0.20
        if idade > 16: prob_evasao += 0.10
        if sexo == "F" and 14 <= idade <= 17: prob_evasao += 0.05  # gravidez na adolescência
        if bolsa_familia: prob_evasao -= 0.08

        evadiu = int(np.random.random() < min(prob_evasao, 0.85))

        motivo_evasao = None
        if evadiu:
            motivos = ["Trabalho", "Gravidez", "Falta de interesse", "Violência", "Distância", "Outros"]
            pesos = [0.35, 0.15, 0.20, 0.10, 0.12, 0.08]
            if sexo == "M": pesos = [0.45, 0.01, 0.22, 0.12, 0.12, 0.08]
            motivo_evasao = np.random.choice(motivos, p=pesos)

        dados.append({
            "id_individuo": f"IND{i+1:06d}",
            "idade": idade,
            "sexo": sexo,
            "raca_cor": raca,
            "renda_per_capita": round(renda, 2),
            "trabalha": trabalha,
            "bolsa_familia": bolsa_familia,
            "evadiu_escola": evadiu,
            "motivo_evasao": motivo_evasao,
            "anos_estudo_mae": np.random.randint(0, 15),
        })

    return pd.DataFrame(dados)


def gerar_dados_municipios(n_municipios: int = 500) -> pd.DataFrame:
    """
    Simula dados municipais: IDH, violência e beneficiários.
    """
    np.random.seed(999)
    regioes = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

    dados = []
    for i in range(n_municipios):
        regiao = np.random.choice(regioes, p=[0.15, 0.30, 0.10, 0.30, 0.15])
        idh_base = {"Norte": 0.62, "Nordeste": 0.61, "Centro-Oeste": 0.73,
                    "Sudeste": 0.76, "Sul": 0.77}[regiao]

        idh = round(np.clip(np.random.normal(idh_base, 0.05), 0.40, 0.95), 3)
        homicidios = max(0, np.random.exponential(18) if regiao in ["Norte", "Nordeste"] else np.random.exponential(10))
        pct_bolsa = round(np.random.uniform(0.05, 0.65) if regiao in ["Norte", "Nordeste"] else np.random.uniform(0.02, 0.30), 3)
        taxa_abandono = round(max(0, (1 - idh) * 30 + np.random.normal(0, 2)), 2)

        dados.append({
            "id_municipio": f"MUN{i+1:04d}",
            "regiao": regiao,
            "populacao": np.random.randint(5000, 2000000),
            "idh": idh,
            "idh_educacao": round(idh * np.random.uniform(0.85, 1.0), 3),
            "taxa_homicidios_100k": round(homicidios, 1),
            "pct_beneficiarios_bolsa": pct_bolsa,
            "taxa_abandono_escolar": taxa_abandono,
            "pct_trabalho_infantil": round(max(0, (1 - idh) * 0.25 + np.random.normal(0, 0.02)), 3),
        })

    return pd.DataFrame(dados)


# ──────────────────────────────────────────────
# Funções de coleta (produção)
# ──────────────────────────────────────────────

def baixar_arquivo(url: str, destino: Path, timeout: int = 120) -> bool:
    """Baixa um arquivo de uma URL para o destino especificado."""
    try:
        log.info(f"Baixando: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(destino, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info(f"✅ Salvo em: {destino} ({destino.stat().st_size / 1024:.1f} KB)")
        return True

    except Exception as e:
        log.warning(f"⚠️ Erro ao baixar {url}: {e}")
        return False


def validar_dataset(df: pd.DataFrame, nome: str) -> dict:
    """Valida qualidade básica do dataset."""
    resultado = {
        "nome": nome,
        "linhas": len(df),
        "colunas": len(df.columns),
        "nulos_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "duplicatas": df.duplicated().sum(),
        "valido": True,
        "alertas": []
    }

    if resultado["nulos_pct"] > 20:
        resultado["alertas"].append(f"Alto percentual de nulos: {resultado['nulos_pct']}%")
    if resultado["duplicatas"] > 0:
        resultado["alertas"].append(f"Duplicatas encontradas: {resultado['duplicatas']}")
    if len(df) < 100:
        resultado["alertas"].append("Dataset muito pequeno")
        resultado["valido"] = False

    return resultado


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("COLETA DE DADOS — EVASÃO ESCOLAR NO BRASIL")
    log.info(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    datasets = {}

    # 1. Censo Escolar
    log.info("\n📊 [1/3] Gerando dados do Censo Escolar (INEP)...")
    df_censo = gerar_dados_censo_escolar(n_escolas=5000)
    caminho = DATA_DIR / "censo_escolar_2023.csv"
    df_censo.to_csv(caminho, index=False, encoding="utf-8")
    datasets["censo_escolar"] = validar_dataset(df_censo, "Censo Escolar")
    log.info(f"✅ {len(df_censo)} escolas salvas em {caminho}")

    # 2. PNAD Contínua
    log.info("\n👥 [2/3] Gerando dados da PNAD Contínua (IBGE)...")
    df_pnad = gerar_dados_pnad(n_individuos=10000)
    caminho = DATA_DIR / "pnad_continua_2023.csv"
    df_pnad.to_csv(caminho, index=False, encoding="utf-8")
    datasets["pnad"] = validar_dataset(df_pnad, "PNAD Contínua")
    log.info(f"✅ {len(df_pnad)} indivíduos salvos em {caminho}")

    # 3. Municípios
    log.info("\n🗺️ [3/3] Gerando dados municipais (IDH + Violência + Bolsa Família)...")
    df_mun = gerar_dados_municipios(n_municipios=500)
    caminho = DATA_DIR / "dados_municipios.csv"
    df_mun.to_csv(caminho, index=False, encoding="utf-8")
    datasets["municipios"] = validar_dataset(df_mun, "Dados Municipais")
    log.info(f"✅ {len(df_mun)} municípios salvos em {caminho}")

    # Relatório de validação
    log.info("\n" + "=" * 60)
    log.info("RELATÓRIO DE VALIDAÇÃO")
    log.info("=" * 60)
    for nome, info in datasets.items():
        status = "✅" if info["valido"] else "❌"
        log.info(f"{status} {info['nome']}: {info['linhas']} linhas | {info['colunas']} colunas | {info['nulos_pct']}% nulos")
        for alerta in info["alertas"]:
            log.warning(f"   ⚠️  {alerta}")

    log.info("\n✅ Coleta finalizada com sucesso!")
    log.info(f"📁 Dados salvos em: {DATA_DIR.resolve()}")

    return datasets


if __name__ == "__main__":
    main()
