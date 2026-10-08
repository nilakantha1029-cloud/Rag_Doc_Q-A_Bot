import re
from collections import Counter
from pathlib import Path
from pypdf import PdfReader

def _norm(line):
    return re.sub(r"\d+", "#", line.strip().lower())

def load_pdfs(folder="data"):
    docs = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        pages = [(p.extract_text() or "") for p in PdfReader(pdf).pages]

        # lines that repeat on many pages are headers/footers
        counts = Counter()
        for text in pages:
            counts.update({_norm(l) for l in text.splitlines() if l.strip()})
        threshold = max(3, int(len(pages) * 0.4))
        repeated = {l for l, c in counts.items() if c >= threshold and len(l) < 120}

        for page_no, text in enumerate(pages, start=1):
            kept = []
            for line in text.splitlines():
                s = line.strip()
                if not s:
                    kept.append("")
                elif _norm(s) in repeated:
                    continue
                elif re.fullmatch(r"(page\s*)?\d{1,3}", s, re.I):
                    continue                      # bare page numbers
                else:
                    kept.append(s)
            clean = "\n".join(kept).strip()
            if clean:
                docs.append({"text": clean, "source": pdf.name, "page": page_no})
    return docs