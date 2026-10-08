import re
from vectorstore import search_diverse
from generator import generate_answer

while True:
    q = input("\nAsk (or 'q'): ").strip()
    if q.lower() == "q":
        break
    if not q:
        continue

    hits = search_diverse(q, k=8)
    answer = generate_answer(q, hits)

    # normalise odd citation styles like 【3†L1-L3】 into [3]
    answer = re.sub(r"【(\d+)[^】]*】", r"[\1]", answer)
    print("\n" + answer)

    # show the sources the model cited, numbered to match
    if "couldn't find" not in answer.lower():
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        if not cited:                          # model skipped [n] citations
            cited = list(range(1, len(hits) + 1))
        print("\nSources:")
        for n in cited:
            if 1 <= n <= len(hits):
                m = hits[n - 1]["meta"]
                print(f"  [{n}] {m['source']}, page {m['page']}")