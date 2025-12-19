import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# -------- LangChain / LangGraph --------
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ----------------- ENV -----------------
load_dotenv()
st.set_page_config(page_title="Smart Agent (FAISS + Memory)", layout="wide")
st.title("🧠 Smart Agent (Memory + PDF + Wikipedia)")

# ----------------- TOOLS -----------------
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------- STATE -----------------
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: List[Document]
    generation: str

class RouterModel(BaseModel):
    datasource: Literal["wiki_search", "vectorstore", "general_chat"] = Field(
        description="Route user query"
    )

# ----------------- VECTOR STORE -----------------
def create_faiss_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# ----------------- NODES -----------------
def retrieve(state: GraphState):
    print("🔍 RETRIEVE")
    retriever = st.session_state.get("retriever")

    if not retriever:
        return {
            "documents": [],
            "question": state["question"],
            "messages": [],
            "generation": ""
        }

    docs = retriever.invoke(state["question"])
    return {
        "documents": docs,
        "question": state["question"],
        "messages": [],
        "generation": ""
    }

def wiki_search(state: GraphState):
    print("🌍 WIKI SEARCH")
    result = wiki_tool.invoke({"query": state["question"]})
    doc = Document(page_content=result, metadata={"source": "wikipedia"})

    return {
        "documents": [doc],
        "question": state["question"],
        "messages": [],
        "generation": ""
    }

def general_chat(state: GraphState):
    print("💬 GENERAL CHAT")
    return {
        "documents": [],
        "question": state["question"],
        "messages": [],
        "generation": ""
    }

def generate(state: GraphState):
    print("✍️ GENERATE")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    if state["documents"]:
        context = "\n\n".join(d.page_content for d in state["documents"])
        system = SystemMessage(
            content=f"You are a helpful assistant. Use the context below:\n\n{context}"
        )
    else:
        system = SystemMessage(content="You are a helpful and friendly assistant.")

    response = llm.invoke([system] + state["messages"])

    return {
        "generation": response.content,
        "messages": [response]
    }

def route_question(state: GraphState):
    print("🧭 ROUTER")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    router_llm = llm.with_structured_output(RouterModel)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Route the question:\n"
         "- PDF related → vectorstore\n"
         "- Factual info → wiki_search\n"
         "- Greetings / chat → general_chat"),
        ("human", "{query}")
    ])

    decision = (prompt | router_llm).invoke({"query": state["question"]})
    return decision.datasource

# ----------------- GRAPH -----------------
def build_graph():
    memory = MemorySaver()
    graph = StateGraph(GraphState)

    graph.add_node("wiki_search", wiki_search)
    graph.add_node("vectorstore", retrieve)
    graph.add_node("general_chat", general_chat)
    graph.add_node("generate", generate)

    graph.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "vectorstore",
            "general_chat": "general_chat",
        }
    )

    graph.add_edge("wiki_search", "generate")
    graph.add_edge("vectorstore", "generate")
    graph.add_edge("general_chat", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=memory)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("🔑 API Key")
    groq_key = st.text_input("Groq API Key", type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    if st.button("🔴 Reset Memory"):
        st.session_state.clear()
        st.rerun()

# ----------------- PDF UPLOAD -----------------
uploaded_pdf = st.file_uploader("Upload PDF (Optional)", type="pdf")

if uploaded_pdf and groq_key:
    pdf_path = f"temp_{uploaded_pdf.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner("📄 Processing PDF..."):
        st.session_state["retriever"] = create_faiss_from_pdf(pdf_path)
        st.success("✅ PDF indexed with FAISS")

# ----------------- GRAPH INIT -----------------
if groq_key and "agent" not in st.session_state:
    st.session_state.agent = build_graph()

# ----------------- CHAT -----------------
query = st.chat_input("Ask anything...")

if query:
    app = st.session_state.agent

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        final = ""

        inputs = {
            "messages": [HumanMessage(content=query)],
            "question": query
        }

        config = {"configurable": {"thread_id": "1"}}

        for step in app.stream(inputs, config=config):
            for k, v in step.items():
                if k == "wiki_search":
                    status.write("🌍 Searching Wikipedia...")
                elif k == "vectorstore":
                    status.write("📂 Searching PDF...")
                elif k == "general_chat":
                    status.write("💬 Chatting...")
                elif k == "generate":
                    final = v["generation"]

        status.update(label="Done", state="complete")
        st.write(final)
