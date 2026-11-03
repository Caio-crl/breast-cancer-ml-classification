"""Gere o relatório acadêmico com Mistune e Chrome, sem executar modelos.

Na raiz do projeto: python scripts/generate_report.py
Requer Mistune e matplotlib no ambiente e Chrome instalado. Não instala pacotes.
"""

import argparse
import ast
import base64
import html
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote

import mistune


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/relatorio_tecnico.md"
PDF = ROOT / "docs/relatorio_tecnico.pdf"
FIGURE = ROOT / "docs/figures/vision_confusion_matrix.png"


def create_figure(source):
    """Represente contagens já documentadas; não carrega checkpoint nem dados."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    section = source.split("## 10. Computer Vision:", 1)[1]
    match = re.search(r"\[\[\d+,\s*\d+\],\s*\[\d+,\s*\d+\]\]", section)
    if match is None:
        raise ValueError("Matriz de Computer Vision ausente no fonte.")
    matrix = ast.literal_eval(match[0])
    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    axis.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=17,
                      color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "#182b40")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benigno (0)", "Maligno (1)"],
             yticklabels=["Benigno (0)", "Maligno (1)"], xlabel="Previsão", ylabel="Diagnóstico real",
             title="MobileNetV3 Small — teste oficial (378 recortes)")
    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=200, metadata={"Software": "Matplotlib"})
    plt.close(figure)


def build_html(source, ref):
    cover_source, body_source = source.split("## 1. Introdução e objetivo", 1)
    title = re.search(r"^# (.+)$", cover_source, re.M)[1]
    subtitle = re.search(r"^\*\*(.+)\*\*$", cover_source, re.M)[1]
    repository = re.search(r"\*\*Repositório:\*\* \[([^]]+)\]", cover_source)[1]
    video = re.search(r"\*\*Vídeo de demonstração:\*\* (.+)", cover_source)[1]
    identity = "\n".join(line for line in cover_source.splitlines()
                         if any(f"**{key}:**" in line for key in ("Instituição", "Autor", "RM")))
    markdown = mistune.create_markdown(plugins=["table"])
    body = markdown("## 1. Introdução e objetivo" + body_source)

    def image_data(match):
        path = (SOURCE.parent / html.unescape(match[1])).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Imagem do relatório inválida: {path}")
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"'

    body = re.sub(r'src="([^"]+)"', image_data, body)

    def external_link(match):
        href = html.unescape(match[1])
        if href.startswith(("http://", "https://", "#")):
            return match[0]
        path = (SOURCE.parent / href).resolve()
        if not path.is_relative_to(ROOT) or not path.exists():
            raise ValueError(f"Referência local ausente: {href}")
        kind = "tree" if path.is_dir() else "blob"
        url = repository.rstrip("/") + f"/{kind}/" + quote(ref, safe="/") + "/" + quote(path.relative_to(ROOT).as_posix())
        return 'href="' + html.escape(url, quote=True) + '"'

    body = re.sub(r'href="([^"]+)"', external_link, body)
    body = re.sub(r"<p>(<img .*?)</p>\s*<p><em>(.*?)</em></p>",
                  r"<figure>\1<figcaption>\2</figcaption></figure>", body, flags=re.S)
    body = body.replace('<h2>10. Computer Vision:', '<h2 class="section-start">10. Computer Vision:')
    body = body.replace('<h2>11. Conclusão e referências', '<h2 class="section-start">11. Conclusão e referências')
    css = """
    @page { size: A4 portrait; margin: 19mm 20mm 19mm;
      @bottom-left { content: 'Tech Challenge — Fase 1 | Relatório técnico acadêmico';
        font-family: Arial, sans-serif; font-size: 7.5pt; color: #666; }
      @bottom-right { content: 'Página ' counter(page) ' de ' counter(pages);
        font-family: Arial, sans-serif; font-size: 7.5pt; color: #666; }
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: #24292f; font: 11.2pt/1.43 Arial, sans-serif;
      -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    h1,h2 { color: #182b40; } h2 { font-size: 15pt; margin: 18pt 0 9pt; }
    h3 { font-size: 12pt; margin: 14pt 0 7pt; } h2,h3 { break-after: avoid; }
    p { margin: 0 0 10pt; orphans: 3; widows: 3; }
    a { color: #245578; text-decoration: underline; overflow-wrap: anywhere; }
    code { font: 9.5pt Consolas, monospace; overflow-wrap: anywhere; }
    table { border-collapse: collapse; width: 100%; margin: 10pt 0 13pt; font-size: 10pt; }
    th,td { padding: 7pt 6pt; border: 1px solid #cbd2d8; vertical-align: top; }
    th { color: #182b40; background: #edf1f5; } tr { break-inside: avoid; }
    thead { display: table-header-group; } table { break-inside: avoid; }
    ul,ol { padding-left: 20pt; } li { margin: 0 0 6pt; orphans: 3; widows: 3; }
    figure { margin: 14pt 0 16pt; break-inside: avoid; text-align: center; }
    figure img { max-width: 100%; max-height: 114mm; width: auto; height: auto; object-fit: contain; }
    figcaption { font-size: 9pt; line-height: 1.4; text-align: left; margin-top: 6pt; color: #444; }
    .cover { height: 258mm; position: relative; break-after: page; padding-top: 13mm; }
    .brand { text-align: center; font-size: 29pt; font-weight: bold; letter-spacing: 1pt; }
    .challenge { text-align: center; font-size: 15pt; margin-top: 13pt; }
    .cover h1 { text-align: center; font-size: 25pt; line-height: 1.25; margin: 60mm 0 12mm; }
    .subtitle { text-align: center; font-size: 12pt; font-weight: bold; }
    .identity { margin: 13mm 15mm 0; } .identity ul { list-style: none; padding: 0; }
    .cover-links { position: absolute; bottom: 13mm; left: 0; right: 0; text-align: center; font-size: 10pt; }
    .section-start { break-before: page; margin-top: 0; }
    """
    return (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            f'<title>{html.escape(title)}</title><style>{css}</style></head><body>'
            f'<section class="cover"><div class="brand">FIAP</div>'
            f'<div class="challenge">Tech Challenge — Fase 1</div><h1>{html.escape(title)}</h1>'
            f'<div class="subtitle">{html.escape(subtitle)}</div><div class="identity">{markdown(identity)}</div>'
            f'<div class="cover-links"><p><a href="{html.escape(repository)}">{html.escape(repository)}</a></p>'
            f'<p>Vídeo de demonstração: <a href="{html.escape(video, quote=True)}">{html.escape(video)}</a></p>'
            f'</div></section><main>{body}</main></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", type=Path, default=Path("C:/Program Files/Google/Chrome/Application/chrome.exe"))
    parser.add_argument("--ref", default="feature/computer-vision", help="Branch ou commit das referências GitHub.")
    args = parser.parse_args()
    if not args.chrome.is_file():
        parser.error("Chrome não encontrado; forneça --chrome com o executável instalado.")
    source = SOURCE.read_text(encoding="utf-8")
    create_figure(source)
    document = build_html(source, args.ref)
    with tempfile.TemporaryDirectory(prefix="fiap_report_print_") as directory:
        temporary = Path(directory)
        html_path = temporary / "report.html"
        html_path.write_text(document, encoding="utf-8")
        output = temporary / "report.pdf"
        result = subprocess.run([str(args.chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                                 "--allow-file-access-from-files", "--no-first-run", "--disable-extensions",
                                 f"--user-data-dir={temporary / 'chrome'}", f"--print-to-pdf={output}",
                                 html_path.as_uri()], capture_output=True, timeout=90)
        if result.returncode or not output.is_file() or not output.read_bytes().startswith(b"%PDF"):
            raise RuntimeError("Chrome não gerou o PDF: " + result.stderr.decode(errors="replace")[-2000:])
        shutil.copyfile(output, PDF)
    print(f"PDF gerado: {PDF}")
    print(f"Figura gerada: {FIGURE}")


if __name__ == "__main__":
    main()
