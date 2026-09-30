import asyncio
import select
import sys
from pathlib import Path

# Garante que a raiz do repositório esteja no PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzers.claim_extractor import ClaimExtractorAnalyzer, format_cli_result



async def main() -> None:
    extractor = ClaimExtractorAnalyzer()

    # 1. Se passou o texto como argumento direto:
    # python scripts/test_claim_extractor.py "URGENTE! O governo aprovou..."
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print("\n🔍 Analisando texto recebido via argumento...")
        result = await extractor.analyze(text, [])
        format_cli_result(result)
        return


    # 2. Modo Interativo Contínuo (Cole a mensagem e pressione ENTER)
    print("=" * 70)
    print("🔎 FactChkBR - Extrator Factual Adaptativo (Terminal Interativo)")
    print("Cole qualquer mensagem ou notícia abaixo e pressione ENTER.")
    print("Para mensagens com múltiplas linhas, cole normalmente.")
    print("Digite 'sair' ou pressione Ctrl+C para encerrar.")
    print("=" * 70)

    while True:
        try:
            print("\n📥 Cole o texto a ser analisado:")
            first_line = input("> ").strip()
            if not first_line:
                continue
            if first_line.lower() in ("sair", "exit", "quit", "q"):
                print("Encerrando testador.")
                break

            lines = [first_line]
            try:
                while select.select([sys.stdin], [], [], 0.05)[0]:
                    extra = sys.stdin.readline()
                    if not extra:
                        break
                    lines.append(extra.strip())
            except Exception:
                pass

            full_text = " ".join(line for line in lines if line)
            result = await extractor.analyze(full_text, [])
            format_cli_result(result)

        except (KeyboardInterrupt, EOFError):
            print("\nSessão encerrada.")
            break


if __name__ == "__main__":
    asyncio.run(main())
