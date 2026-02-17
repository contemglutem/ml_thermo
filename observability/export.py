"""Exportação de resultados."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .metrics import ModelMetrics


def save_results(
        data: Dict[str, Any],
        metrics: ModelMetrics,
        output_dir: str,
        experiment_name: str,
) -> None:
    """
    Salva todos os resultados do experimento.

    Args:
        data: Dicionário com dados do experimento
        metrics: Métricas calculadas
        output_dir: Diretório de saída
        experiment_name: Nome do experimento
    """
    output_path = Path(output_dir)

    # Salvar como JSON
    export_to_json(
        data=data,
        metrics=metrics,
        filepath=output_path / "reports" / f"{experiment_name}.json"
    )

    # Salvar como CSV
    export_to_csv(
        data=data,
        filepath=output_path / "reports" / f"{experiment_name}.csv"
    )


def export_to_json(
        data: Dict[str, Any],
        metrics: ModelMetrics,
        filepath: str,
) -> None:
    """
    Exporta resultados para JSON.

    Args:
        data: Dados do experimento
        metrics: Métricas
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Converter numpy arrays para listas
    json_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            json_data[key] = value.tolist()
        elif value is not None:
            json_data[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value

    # Adicionar métricas
    json_data['metrics'] = metrics.to_dict()

    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ Resultados exportados para {filepath}")


def export_to_csv(
        data: Dict[str, Any],
        filepath: str,
) -> None:
    """
    Exporta dados para CSV.

    Args:
        data: Dados do experimento
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Criar DataFrame com dados principais
    df_data = {}

    # Adicionar arrays que têm o mesmo tamanho
    for key in ['y_positions', 'T_true', 'T_pinn', 'T_analytical']:
        if key in data and data[key] is not None:
            df_data[key] = data[key]

    if df_data:
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        print(f"✓ Dados exportados para {filepath}")