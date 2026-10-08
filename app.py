import os, re, sys, time, html

# run from the project root no matter where streamlit is launched
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import streamlit as st

st.set_page_config(page_title="Paperlight", page_icon="🔦", layout="wide",initial_sidebar_state="expanded",)

from vectorstore import collection, search_diverse
from generator import generate_answer

# ---------- styling ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Newsreader:ital,wght@0,400;0,500;1,400&display=swap');

:root { --ink:#0d1320; --ink2:#141d30; --beam:#5eead4; --amber:#ffc857; --text:#e7ecf5; --mute:#8fa0bd; }

.stApp {
  background: radial-gradient(1200px 600px at 15% -10%, #1b2a4a 0%, transparent 60%),
              radial-gradient(900px 500px at 100% 110%, #10353a 0%, transparent 55%),
              var(--ink);
  color: var(--text);
  font-family: 'Newsreader', Georgia, serif;
}
/* keep the header (it holds the sidebar toggle), just make it invisible-looking */
header[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu,
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden; }

/* make the sidebar open/close arrows clearly visible on the dark theme */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] * { color: var(--beam) !important; }

/* hero */
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(2.2rem, 6vw, 3.6rem); letter-spacing: -0.03em; margin: 0; line-height: 1.02;
  background: linear-gradient(100deg, #e7ecf5 0%, #e7ecf5 35%, var(--beam) 50%, #e7ecf5 65%, #e7ecf5 100%);
  background-size: 250% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: sheen 7s ease-in-out infinite;
}
.hero p { color: var(--mute); font-size: 1.15rem; margin: .6rem 0 0; max-width: 34em; }
@keyframes sheen { 0%,15% { background-position: 100% 0; } 55%,100% { background-position: 0% 0; } }

/* the scan beam: runs while the bot reads the papers */
.scan { position: relative; height: 54px; border-radius: 10px; overflow: hidden;
  background: repeating-linear-gradient(0deg, #ffffff10 0 1px, transparent 1px 9px), var(--ink2);
  border: 1px solid #ffffff14; }
.scan::after { content: ""; position: absolute; top: 0; bottom: 0; width: 90px; left: -90px;
  background: linear-gradient(90deg, transparent, #5eead455, var(--beam), #5eead455, transparent);
  animation: sweep 1.3s linear infinite; }
.scan span { position: absolute; left: 14px; top: 50%; transform: translateY(-50%);
  font-family: 'Bricolage Grotesque', sans-serif; font-size: .95rem; color: var(--text); z-index: 2; }
@keyframes sweep { to { left: 100%; } }

/* citations */
.cite { display: inline-block; min-width: 1.35em; padding: 0 .35em; margin: 0 .2em; text-align: center;
  font: 700 .78rem 'Bricolage Grotesque', sans-serif; line-height: 1.5; color: #1a1400;
  background: var(--amber); border-radius: 4px; vertical-align: .12em; }

/* source slips */
.srcs { display: flex; flex-direction: column; gap: .5rem; margin-top: 1rem; }
.src { background: var(--ink2); border: 1px solid #ffffff14; border-left: 3px solid var(--amber);
  border-radius: 8px; padding: .55rem .8rem; animation: lit .9s ease-out both; }
.src summary { cursor: pointer; display: flex; align-items: center; gap: .6rem; list-style: none;
  font-family: 'Bricolage Grotesque', sans-serif; font-size: .92rem; }
.src summary::-webkit-details-marker { display: none; }
.src summary em { color: var(--mute); font-style: normal; margin-left: auto; }
.src p { color: #b9c5da; font-size: .98rem; line-height: 1.55; margin: .6rem 0 .2rem; }
.src[open] { border-left-color: var(--beam); }
@keyframes lit { 0% { box-shadow: 0 0 0 0 #ffc85700; border-color: #ffc857; background: #2a2412; }
                 100% { box-shadow: 0 0 0 0 #ffc85700; } }

/* document spines in the sidebar */
section[data-testid="stSidebar"] { background: #0a0f1a; border-right: 1px solid #ffffff10; }
.side-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; margin: .2rem 0 .2rem; }
.side-sub { color: var(--mute); font-size: .95rem; margin-bottom: 1rem; }
.spine { padding: .7rem .8rem; margin-bottom: .55rem; border-radius: 6px 10px 10px 6px;
  background: var(--ink2); border-left: 5px solid #34415f; transition: all .5s ease; }
.spine b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: .9rem; font-weight: 500; }
.spine span { color: var(--mute); font-size: .85rem; }
.spine.lit { border-left-color: var(--beam); background: #12302f; box-shadow: 0 0 22px #5eead433;
  transform: translateX(5px); }

/* chat bubbles + input */
[data-testid="stChatMessage"] { background: transparent; padding: .4rem 0; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 1.12rem; line-height: 1.65; }
[data-testid="stChatInput"] { border: 1px solid #ffffff1f; border-radius: 14px; background: var(--ink2); }
[data-testid="stChatInput"]:focus-within { border-color: var(--beam); box-shadow: 0 0 0 3px #5eead422; }

/* suggestion chips */
.stButton > button { width: 100%; text-align: left; background: var(--ink2); color: var(--text);
  border: 1px solid #ffffff1a; border-radius: 12px; padding: .75rem .9rem; font-size: 1rem;
  font-family: 'Newsreader', serif; transition: border-color .2s, background .2s; }
.stButton > button:hover { border-color: var(--beam); background: #12302f; color: var(--text); }

@media (prefers-reduced-motion: reduce) {
  .hero h1, .scan::after, .src { animation: none !important; }
  .spine { transition: none; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- data ----------
@st.cache_data
def doc_stats():
    metas = collection.get(include=["metadatas"])["metadatas"]
    pages = {}
    for m in metas:
        pages.setdefault(m["source"], set()).add(m["page"])
    return {k: len(v) for k, v in sorted(pages.items())}, collection.count()

docs, total_chunks = doc_stats()

SUGGESTED = [
    "How many users did ChatGPT reach in its first two months?",
    "What is the 5% problem in AI and education?",
    "Do the documents agree on whether AI will replace jobs?",
    "Which jobs are most resistant to AI?",
]

def pretty(name):
    return re.sub(r"[_]+", " ", name.rsplit(".", 1)[0])

def side_html(lit):
    items = ""
    for name, n in docs.items():
        cls = "spine lit" if name in lit else "spine"
        items += f'<div class="{cls}"><b>{html.escape(pretty(name))}</b><span>{n} pages</span></div>'
    return (f'<div class="side-title">Your library</div>'
            f'<div class="side-sub">{len(docs)} papers, {total_chunks} passages. '
            f'Papers glow when an answer uses them.</div>{items}')

def scan_html(text):
    return f'<div class="scan"><span>{html.escape(text)}</span></div>'

def clean(text):
    return re.sub(r"【(\d+)[^】]*】", r"[\1]", text)

def with_badges(text):
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', text)

def sources_html(sources):
    cards = ""
    for i, s in enumerate(sources):
        snippet = html.escape(s["text"][:900]).replace("\n", " ")
        cards += (f'<details class="src" style="animation-delay:{i * 0.18}s"><summary>'
                  f'<span class="cite">{s["n"]}</span><b>{html.escape(pretty(s["source"]))}</b>'
                  f'<em>page {s["page"]}</em></summary><p>{snippet}…</p></details>')
    return f'<div class="srcs">{cards}</div>'

# ---------- state ----------
st.session_state.setdefault("history", [])
st.session_state.setdefault("lit", set())

side = st.sidebar.empty()
side.markdown(side_html(st.session_state.lit), unsafe_allow_html=True)
if st.sidebar.button("Clear conversation"):
    st.session_state.history, st.session_state.lit = [], set()
    st.rerun()

# ---------- hero ----------
st.markdown(
    '<div class="hero"><h1>Paperlight</h1>'
    '<p>Ask your four AI papers a question. Every answer points to the page it came from.</p></div>',
    unsafe_allow_html=True)
st.write("")

# ---------- past messages ----------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            st.markdown(with_badges(m["content"]), unsafe_allow_html=True)
            if m["sources"]:
                st.markdown(sources_html(m["sources"]), unsafe_allow_html=True)

# ---------- input ----------
typed = st.chat_input("Ask about the papers…")   # always render the box
q = st.session_state.pop("pending", None) or typed

if not st.session_state.history and not q:
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTED):
        if cols[i % 2].button(s, key=f"sg{i}"):
            st.session_state.pending = s
            st.rerun()

# ---------- answer ----------
if q:
    st.session_state.history.append({"role": "user", "content": q, "sources": []})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        ph = st.empty()
        try:
            ph.markdown(scan_html(f"Searching {total_chunks} passages…"), unsafe_allow_html=True)
            hits = search_diverse(q, k=8)
            ph.markdown(scan_html("Reading the closest matches…"), unsafe_allow_html=True)
            raw = clean(generate_answer(q, hits))
        except Exception as e:
            ph.error(f"Could not get an answer: {e}")
            st.stop()

        # type the answer out word by word
        shown = ""
        for w in raw.split(" "):
            shown += w + " "
            ph.markdown(shown + "▌")
            time.sleep(0.015)
        ph.markdown(with_badges(raw), unsafe_allow_html=True)

        sources = []
        if "couldn't find" not in raw.lower():
            cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", raw)})
            if not cited:                      # model skipped [n] citations
                cited = list(range(1, len(hits) + 1))
            for n in cited:
                if 1 <= n <= len(hits):
                    h = hits[n - 1]
                    sources.append({"n": n, "source": h["meta"]["source"],
                                    "page": h["meta"]["page"], "text": h["text"]})
        if sources:
            st.markdown(sources_html(sources), unsafe_allow_html=True)

    lit = {s["source"] for s in sources}
    st.session_state.lit = lit
    side.markdown(side_html(lit), unsafe_allow_html=True)
    st.session_state.history.append({"role": "assistant", "content": raw, "sources": sources})