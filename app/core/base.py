from abc import ABC, abstractmethod
from app.schemas.analysis import AnalyzerResult


class BaseAnalyzer(ABC):
    """
    Interface abstrata para analisadores de verificação de fatos.
    Todo novo modelo, heurística ou integração externa deve herdar desta classe.
    """

    name: str = "base"
    weight: float = 1.0

    @abstractmethod
    async def analyze(self, text: str, urls: list[str]) -> AnalyzerResult:
        """
        Executa a análise de veracidade sobre o texto e URLs fornecidas.

        :param text: Texto da mensagem ou notícia a ser verificada.
        :param urls: Lista de URLs anexadas ou mencionadas.
        :return: AnalyzerResult contendo o veredito parcial, confiança, razões e fontes.
        """
        raise NotImplementedError("Subclasses devem implementar o método analyze.")
