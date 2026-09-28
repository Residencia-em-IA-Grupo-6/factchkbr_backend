import importlib
import pkgutil
from typing import Callable
from app.core.base import BaseAnalyzer


class AnalyzerRegistry:
    """
    Registro dinâmico de analisadores (Registry Pattern).
    Permite registrar novos modelos sem alterar o código do orquestrador.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseAnalyzer]] = {}
        self._weights: dict[str, float] = {}

    def register(self, name: str, weight: float = 1.0) -> Callable[[type[BaseAnalyzer]], type[BaseAnalyzer]]:
        """
        Decorator para registrar um novo analisador no sistema.

        Exemplo:
            @register_analyzer("meu_modelo", weight=1.5)
            class MeuModelo(BaseAnalyzer):
                ...
        """
        def decorator(cls: type[BaseAnalyzer]) -> type[BaseAnalyzer]:
            cls.name = name
            cls.weight = weight
            self._registry[name] = cls
            self._weights[name] = weight
            return cls
        return decorator

    def get(self, name: str) -> type[BaseAnalyzer] | None:
        """Retorna a classe do analisador registrado."""
        return self._registry.get(name)

    def list_available(self) -> list[str]:
        """Lista os nomes de todos os analisadores registrados."""
        return sorted(list(self._registry.keys()))

    def auto_discover(self, package_name: str = "app.analyzers") -> None:
        """
        Importa automaticamente os módulos do pacote para ativar os decoradores.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return

        if not hasattr(package, "__path__"):
            return

        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package_name}.{module_name}")


# Instância global do Registry
registry = AnalyzerRegistry()

# Decorator para exportação direta
register_analyzer = registry.register
