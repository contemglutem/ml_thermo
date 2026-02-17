"""Configuração para ambiente Docker/headless."""

import os
import matplotlib

# Detectar se está rodando em ambiente headless (Docker, servidor, etc)
def is_headless_environment() -> bool:
    """Detecta se está em ambiente sem display."""
    return (
        os.environ.get('DISPLAY') is None or
        os.environ.get('DOCKER_CONTAINER') == 'true' or
        os.environ.get('CI') == 'true'
    )


def configure_matplotlib_backend():
    """Configura backend apropriado do matplotlib."""
    if is_headless_environment():
        # Usar backend 'Agg' para salvar sem display
        matplotlib.use('Agg')
        print("🐳 Modo headless detectado - usando backend 'Agg'")
    else:
        # Ambiente normal com display
        print("🖥️  Display disponível - usando backend padrão")


# Configurar automaticamente na importação
configure_matplotlib_backend()
