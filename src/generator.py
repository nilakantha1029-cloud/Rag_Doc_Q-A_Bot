import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-120b"  

SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the context answers only part of the question, answer that part and say plainly "
    "which part the documents do not cover. Use the exact sentence "
    "'I couldn't find that in the documents.' only when nothing relevant is in the context. "
    "Cite sources using plain square brackets with the context number, like [1] or [2]. "
    "Every claim must end with a bracketed context number. "
    "Never cite by file name or page alone, and do not use any other citation format. "
    "Keep answers concise. "
    "When the question compares sources or asks whether they agree, answer in exactly this shape: "
    "(1) Where the sources agree. "
    "(2) Where they differ in emphasis, naming each file with its number, like 'Ai Replace human jobs [1]'. "
    "(3) One-sentence takeaway that restates (1) and (2) in plain words. "
    "Do not use the words 'contradict' or 'disagree'. "
    "Describe differences as differences in emphasis or focus. "
    "If the question assumes something the context contradicts, correct it using the context. "
    "If a source makes a claim and also lists caveats or opposing points, report both. "
)

def generate_answer(question, contexts):
    context_block = "\n\n".join(
        f"[{i+1}] ({c['meta']['source']}, page {c['meta']['page']})\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,          # keeps answers factual and consistent
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content