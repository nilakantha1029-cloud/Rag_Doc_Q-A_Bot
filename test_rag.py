"""Run from the project root:  python test_rag.py
Asks the bot a fixed set of questions and checks each answer against facts from your 4 PDFs."""
import os, re, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from vectorstore import search_diverse
from generator import generate_answer

REFUSE = ["couldn't find", "could not find", "not cover", "does not mention", "do not mention"]

# t=type  q=question  all=list of groups (every group needs one of its words in the answer)
# files=file-name parts that must be cited   min_files=min distinct PDFs cited   forbid=words that must NOT appear
CASES = [
    # ---- single facts ----
    dict(t="fact", q="Which AI systems beat a chess grandmaster in 1997 and won Jeopardy in 2011?",
         all=[["deep blue"], ["watson"]], files=["wttc"]),
    dict(t="fact", q="What did the Swedish study find about AI breast cancer screening?",
         all=[["50%", "half"], ["20%"]], files=["wttc"]),
    dict(t="fact", q="What market share does TSMC have in chip manufacturing?",
         all=[["55"]], files=["wttc"]),
    dict(t="fact", q="How many GPU chips were used in the supercomputer built to train ChatGPT?",
         all=[["10,000"]], files=["wttc"]),
    dict(t="fact", q="How many people have never been connected to the internet according to the ITU?",
         all=[["2.6 billion"]], files=["wttc"]),
    dict(t="fact", q="How much did the AI painting Edmond de Belamy sell for?",
         all=[["432,500"]], files=["wttc"]),
    dict(t="fact", q="How many children are out of school globally and how many more teachers are needed by 2030?",
         all=[["272 million"], ["44 million"]], files=["future"]),
    dict(t="fact", q="In the challenges table, what percentage is job displacement and what is high implementation cost?",
         all=[["35"], ["20"]], files=["replace"]),
    dict(t="fact", q="Which AI model did Google say outperformed competitors on learning science principles?",
         all=[["gemini 2.5 pro", "gemini 2.5-pro"]], files=["future"]),
    # ---- concepts ----
    dict(t="concept", q="What is explainable AI and why does it matter?",
         all=[["black box", "glass box"], ["trust"]], files=["wttc"]),
    dict(t="concept", q="What is the 5% problem in AI and education?",
         all=[["motivated"]], files=["future"]),
    dict(t="concept", q="What is the difference between a data warehouse and a data lake?",
         all=[["structured"], ["unstructured"]], files=["wttc"]),
    dict(t="concept", q="Why are GPUs well suited to AI?",
         all=[["parallel"]], files=["wttc"]),
    # ---- cross-document ----
    dict(t="cross", q="What do the documents say about AI and creativity?",
         all=[["creativ"]], min_files=2),
    dict(t="cross", q="What do the documents say about hallucinations and accuracy?",
         all=[["hallucinat"]], files=["wttc", "future"]),
    dict(t="cross", q="How is AI used in healthcare across the documents?",
         all=[["cancer", "diagnos", "medical"]], min_files=2),
    dict(t="cross", q="What do the documents say about privacy and data security?",
         all=[["privacy"]], min_files=2),
    dict(t="cross", q="What do the documents say about AI training and reskilling?",
         all=[["reskill", "upskill", "training"]], min_files=2),
    dict(t="cross", q="What do the documents say about inequality or the digital divide?",
         all=[["divide", "inequal", "access"]], min_files=2),
    dict(t="cross", q="What do the documents say about bias in AI?",
         all=[["bias"]], min_files=2),
    dict(t="cross", q="Do the documents agree on whether AI will replace jobs?",
         all=[["transform"], ["displac"], ["uncertain", "unpredictab", "unclear", "not uniform"]],
         min_files=3, forbid=["contradict", "disagree"]),
    # ---- tricky ----
    dict(t="tricky", q="In what year was ChatGPT released, according to the WTTC report?",
         all=[["2022"]], files=["wttc"]),
     dict(t="tricky", q="Why did the WTTC report say ChatGPT launched in 2019?",
         all=[["2022", "couldn't find", "could not find", "does not say", "not mention"]]),                      
    dict(t="tricky", q="Does the pros-and-cons paper say AI is unbiased?",
         all=[["bias"]], files=["cones"]),
    dict(t="tricky", q="Will AI replace teachers according to the documents?",
         all=[["teacher", "educat"]]),
    # ---- not in the PDFs: must refuse ----
    dict(t="refuse", q="What is the price of a GPT-5 subscription?", refuse=True),
    dict(t="refuse", q="What does the document say about quantum computing?", refuse=True),
    dict(t="refuse", q="Who is the CEO of Anthropic?", refuse=True),
    dict(t="refuse", q="What is the weather in Hyderabad today?", refuse=True),
    dict(t="refuse", q="What did Google report as its revenue in 2025?", refuse=True),
]


def norm(s):
    s = s.lower().replace("**", "")
    s = s.replace("\u202f", " ").replace("\xa0", " ")
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)      # fancy hyphens -> "-"
    s = re.sub(r"\s*%", "%", s)                        # "20 %" -> "20%"
    s = re.sub(r"\s*(percent|per cent)\b", "%", s)     # "20 percent" -> "20%"
    return s


def ask(q):
    hits = search_diverse(q, k=8)
    raw = generate_answer(q, hits)
    raw = re.sub(r"【(\d+)[^】]*】", r"[\1]", raw)
    return hits, raw


def cited_files(raw, hits):
    nums = sorted({int(x) for x in re.findall(r"\[(\d+)\]", raw)})
    chosen = [hits[n - 1] for n in nums if 1 <= n <= len(hits)] or hits
    return {h["meta"]["source"].lower() for h in chosen}


def has_file(parts, files):
    return all(any(p in f for f in files) for p in parts)


def run_case(c):
    for attempt in range(2):
        try:
            hits, raw = ask(c["q"])
            break
        except Exception as e:                      # rate limit etc.
            if attempt == 1:
                return False, f"ERROR {e}", ""
            time.sleep(10)
    text = norm(raw)

    if c.get("refuse"):
        ok = any(p in text for p in REFUSE)
        return ok, "" if ok else "answered instead of refusing", raw

    retrieved = {h["meta"]["source"].lower() for h in hits}
    cited = cited_files(raw, hits)
    problems = []
    for grp in c.get("all", []):
        if not any(w in text for w in grp):
            problems.append(f"missing {grp}")
    for w in c.get("forbid", []):
        if w in text:
            problems.append(f"contains '{w}'")
    if not has_file(c.get("files", []), cited):
        why = "retrieval" if not has_file(c.get("files", []), retrieved) else "citation"
        problems.append(f"expected file not cited ({why})")
    if len(cited) < c.get("min_files", 0):
        problems.append(f"only {len(cited)} PDF(s) cited, need {c['min_files']}")
    return not problems, "; ".join(problems), raw


def main():
    results, log = [], []
    for i, c in enumerate(CASES, 1):
        ok, why, raw = run_case(c)
        results.append((c["t"], ok))
        print(f"{'PASS' if ok else 'FAIL'}  [{c['t']:7}] {c['q']}" + ("" if ok else f"\n        -> {why}"))
        log.append(f"Q{i} [{c['t']}] {c['q']}\n{'PASS' if ok else 'FAIL: ' + why}\n{raw}\n{'-' * 70}\n")
        time.sleep(3)                               # stay under Groq rate limits

    print("\n=== Summary ===")
    for t in ["fact", "concept", "cross", "tricky", "refuse"]:
        r = [ok for tt, ok in results if tt == t]
        print(f"{t:8} {sum(r)}/{len(r)}")
    total = sum(ok for _, ok in results)
    print(f"{'TOTAL':8} {total}/{len(results)}  ({100 * total // len(results)}%)")
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("Full answers saved to test_results.txt")


if __name__ == "__main__":
    main()