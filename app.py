import sys
try:  # needed on Streamlit Cloud so ChromaDB gets a modern sqlite
    _import_("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import datetime
import chromadb
import streamlit as st
from pypdf import PdfReader
from anthropic import Anthropic

st.set_page_config(page_title="StudyBuddy AI")
st.title("StudyBuddy AI")
st.caption("A GenAI study assistant: LLM API + Prompt Engineering + RAG + Tools + Agent")

# ---------- Sidebar ----------
api_key = st.sidebar.text_input("Anthropic API key", type="password")

# Prompt engineering: each mode is a different system prompt
MODES = {
    "Explain simply": "You are a friendly tutor. Explain ideas in simple words with one everyday example. Keep answers short.",
    "Quiz me": "You are a quiz master. Ask one question at a time based on the topic or notes, wait for the answer, then give feedback.",
    "Exam answer": "You are an exam coach. Give structured answers: definition, key points as a short list, and a one-line summary.",
}
mode = st.sidebar.selectbox("Study mode", list(MODES))

notes = st.sidebar.file_uploader("Upload your notes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
if st.sidebar.button("Process notes") and notes:
    # RAG: Document -> Chunking -> Embeddings -> Vector DB
    client_db = chromadb.Client()
    try:
        client_db.delete_collection("notes")
    except Exception:
        pass
    col = client_db.create_collection("notes")
    ids, docs = [], []
    for f in notes:
        if f.name.lower().endswith(".pdf"):
            text = "\n".join(p.extract_text() or "" for p in PdfReader(f).pages)
        else:
            text = f.read().decode("utf-8", errors="ignore")
        for i in range(0, len(text), 700):
            piece = text[i:i + 800]
            if piece.strip():
                ids.append(f"{f.name}-{i}")
                docs.append(piece)
    col.add(ids=ids, documents=docs)
    st.session_state["col"] = col
    st.sidebar.success(f"Stored {len(docs)} chunks")

if st.sidebar.button("Clear chat"):
    st.session_state.pop("messages", None)
    st.session_state.pop("log", None)


# ---------- Tools ----------
def search_notes(query):
    if "col" not in st.session_state:
        return "No notes uploaded yet."
    res = st.session_state["col"].query(query_texts=[query], n_results=3)
    return "\n---\n".join(res["documents"][0])


def calculator(expression):
    if not set(expression) <= set("0123456789+-*/(). %"):
        return "Error: only numbers and + - * / ( ) are allowed"
    try:
        return str(eval(expression, {"_builtins_": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def get_time():
    return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")


def run_tool(name, args):
    if name == "search_notes":
        return search_notes(args["query"])
    if name == "calculator":
        return calculator(args["expression"])
    return get_time()


TOOLS = [
    {"name": "search_notes", "description": "Search the student's uploaded notes. Use for any question about their notes.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculator", "description": "Do math such as 12*(3+4).",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "get_time", "description": "Get the current date and time (for study planning).",
     "input_schema": {"type": "object", "properties": {}}},
]

# ---------- Agent with memory ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("log", [])

question = st.chat_input("Ask a question, e.g. 'Summarise my notes' or 'Quiz me on chapter 1'")

if question:
    if not api_key:
        st.warning("Add your API key in the sidebar.")
    else:
        client = Anthropic(api_key=api_key)
        msgs = st.session_state["messages"]
        msgs.append({"role": "user", "content": question})
        steps = ["Understand task: " + question]
        final = "Stopped: too many steps."
        for _ in range(6):
            reply = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=900,
                system=MODES[mode] + " Use tools when they help.",
                tools=TOOLS, messages=msgs)
            msgs.append({"role": "assistant", "content": reply.content})
            if reply.stop_reason != "tool_use":
                final = "".join(b.text for b in reply.content if b.type == "text")
                steps.append("Respond")
                break
            results = []
            for b in reply.content:
                if b.type == "tool_use":
                    steps.append(f"Decide action: use {b.name} {b.input}")
                    out = run_tool(b.name, b.input)
                    steps.append(f"Process result: {out[:150]}")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            msgs.append({"role": "user", "content": results})
        st.session_state["log"].append({"q": question, "steps": steps, "a": final})

for item in st.session_state["log"]:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        with st.expander("Agent steps"):
            for s in item["steps"]:
                st.write(s)
        st.write(item["a"])
