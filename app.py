import os
import streamlit as st
from pydantic import BaseModel, Field
from typing import Type
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

# ==========================================
# 1. SETUP E CONNESSIONI CLOUD
# ==========================================
st.set_page_config(page_title="AI PM Digital Colleague", page_icon="🚀", layout="wide")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))

if not GOOGLE_API_KEY or not QDRANT_URL:
    st.error("⚠️ Chiavi API mancanti. Configura i Secrets su Streamlit Cloud.")
    st.stop()

# CrewAI moderno usa LiteLLM sotto il cofano, che cerca questa variabile:
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

agente_llm = LLM(
    model="gemini/gemini-1.5-flash-latest",
    api_key=GOOGLE_API_KEY,
    temperature=0.1
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = "pm_knowledge_base"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    return client, collection_name

qdrant_client, collection_name = get_qdrant_client()
vectorstore = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)


# ==========================================
# 2. DEFINIZIONE TOOLS DI MEMORIA (STRUTTURA RIGIDA)
# ==========================================
class SalvaConoscenzaInput(BaseModel):
    testo: str = Field(..., description="Il testo dell'informazione da salvare.")
    categoria: str = Field(..., description="La categoria (es. 'rischi').")

class SalvaConoscenzaTool(BaseTool):
    name: str = "salva_conoscenza"
    description: str = "Salva una nuova informazione nel database."
    args_schema: Type[BaseModel] = SalvaConoscenzaInput

    def _run(self, testo: str, categoria: str) -> str:
        doc = Document(page_content=testo, metadata={"tipo": categoria.lower()})
        vectorstore.add_documents([doc])
        return f"Fatto. Informazione salvata in: {categoria.upper()}."

class RicercaConoscenzaInput(BaseModel):
    query: str = Field(..., description="La domanda da cercare nel database.")

class RicercaConoscenzaTool(BaseTool):
    name: str = "ricerca_conoscenza"
    description: str = "Ricerca semantica nel database."
    args_schema: Type[BaseModel] = RicercaConoscenzaInput

    def _run(self, query: str) -> str:
        docs = vectorstore.similarity_search(query, k=6)
        if not docs:
            return "Nessuna informazione trovata."
        return "\n\n".join([f"[{d.metadata.get('tipo', 'generico').upper()}]: {d.page_content}" for d in docs])

class EsploraMemoriaInput(BaseModel):
    categoria_specifica: str = Field(..., description="Passa 'TUTTE' per l'elenco.")

class EsploraMemoriaTool(BaseTool):
    name: str = "esplora_memoria"
    description: str = "Elenca le categorie o mostra una categoria specifica."
    args_schema: Type[BaseModel] = EsploraMemoriaInput

    def _run(self, categoria_specifica: str) -> str:
        records, _ = qdrant_client.scroll(collection_name=collection_name, limit=200, with_payload=True, with_vectors=False)
        if not records: return "Database vuoto."
        mappatura = {}
        for r in records:
            cat = r.payload.get("metadata", {}).get("tipo", "generale")
            testo = r.payload.get("page_content", "Senza testo")
            mappatura.setdefault(cat, []).append(testo)
        if categoria_specifica.upper() != "TUTTE":
            cat_lower = categoria_specifica.lower()
            if cat_lower in mappatura:
                return f"Contenuto {cat_lower.upper()}:\n" + "\n".join([f"- {t}" for t in mappatura[cat_lower]])
            return f"Categoria non trovata."
        return "Categorie:\n" + "\n".join([f"- {c.upper()}" for c in mappatura.keys()])

class EliminaConoscenzaInput(BaseModel):
    vecchia_informazione: str = Field(..., description="Dato da cancellare.")

class EliminaConoscenzaTool(BaseTool):
    name: str = "elimina_conoscenza"
    description: str = "Cancella un dato obsoleto dal database."
    args_schema: Type[BaseModel] = EliminaConoscenzaInput

    def _run(self, vecchia_informazione: str) -> str:
        vettore_query = embeddings.embed_query(vecchia_informazione)
        risultati = qdrant_client.search(collection_name=collection_name, query_vector=vettore_query, limit=1)
        if risultati:
            qdrant_client.delete(collection_name=collection_name, points_selector=[risultati[0].id])
            return "Vecchio dato eliminato."
        return "Nessun dato trovato."

# ==========================================
# 3. INTERFACCIA UTENTE WEB (STREAMLIT)
# ==========================================
st.title("🤖 Digital PM Colleague")

tab1, tab2 = st.tabs(["🧠 Memoria", "🚀 Pianificazione"])

with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    user_input = st.chat_input("Scrivi qui...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        input_pulito = user_input.replace('\n', ' | ')
        
        agente_architetto = Agent(
            role='Memory Manager',
            goal='Gestire la memoria. Esplora, salva, cerca e correggi.',
            backstory='Architetto dei dati.',
            tools=[SalvaConoscenzaTool(), RicercaConoscenzaTool(), EsploraMemoriaTool(), EliminaConoscenzaTool()],
            llm=agente_llm,
            max_iter=5,
            verbose=True
        )
        
        task_addestramento = Task(
            description=f'''Analizza: "{input_pulito}". Se è un saluto, rispondi e basta. Altrimenti usa i tool per salvare, cercare o esplorare.''',
            expected_output='Risposta operativa.',
            agent=agente_architetto
        )
        
        with st.spinner("Accesso alla memoria..."):
            try:
                risposta = Crew(agents=[agente_architetto], tasks=[task_addestramento]).kickoff()
                testo_risposta = risposta if isinstance(risposta, str) else getattr(risposta, 'raw', str(risposta))
                st.session_state.chat_history.append({"role": "assistant", "content": testo_risposta})
                st.chat_message("assistant").write(testo_risposta)
            except Exception as e:
                st.error(f"Errore: {e}")

with tab2:
    with st.form("project_form"):
        nome_progetto = st.text_input("Progetto")
        impatti = st.text_area("Impatti")
        submit_btn = st.form_submit_button("Genera Piano")
        
    if submit_btn and nome_progetto:
        analista = Agent(role='Analyst', goal='Analizzare impatti.', backstory='Tecnico.', tools=[RicercaConoscenzaTool()], llm=agente_llm)
        planner = Agent(role='Planner', goal='Creare WBS.', backstory='Agile Coach.', llm=agente_llm)
        
        t1 = Task(description=f'Cerca nel DB i referenti per {impatti}.', expected_output='Lista referenti.', agent=analista)
        t2 = Task(description='Crea WBS Agile.', expected_output='Piano WBS.', agent=planner)
        
        with st.spinner("Pianificazione..."):
            risultato = Crew(agents=[analista, planner], tasks=[t1, t2], process=Process.sequential).kickoff()
            st.markdown(getattr(risultato, 'raw', str(risultato)))
