import os
import streamlit as st
from pydantic import BaseModel, Field
from typing import Type
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

# Impostiamo le chiavi necessarie sia per Langchain che per LiteLLM (CrewAI)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# Motore LLM nativo di CrewAI (risolve i ValidationError)
agente_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.2
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

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
    categoria: str = Field(..., description="La categoria (es. 'rischi', 'stack_tecnologico').")

class SalvaConoscenzaTool(BaseTool):
    name: str = "salva_conoscenza"
    description: str = "Salva una nuova informazione nel database creando o usando categorie specifiche."
    args_schema: Type[BaseModel] = SalvaConoscenzaInput

    def _run(self, testo: str, categoria: str) -> str:
        doc = Document(page_content=testo, metadata={"tipo": categoria.lower()})
        vectorstore.add_documents([doc])
        return f"Fatto. Informazione salvata nella categoria: {categoria.upper()}."

class RicercaConoscenzaInput(BaseModel):
    query: str = Field(..., description="La domanda o le parole chiave da cercare nel database.")

class RicercaConoscenzaTool(BaseTool):
    name: str = "ricerca_conoscenza"
    description: str = "Ricerca semantica: usa questo tool per rispondere a domande specifiche cercando nel database."
    args_schema: Type[BaseModel] = RicercaConoscenzaInput

    def _run(self, query: str) -> str:
        docs = vectorstore.similarity_search(query, k=6)
        if not docs:
            return "Non ho trovato informazioni in memoria a riguardo."
        risultati = [f"[{d.metadata.get('tipo', 'generico').upper()}]: {d.page_content}" for d in docs]
        return "\n\n".join(risultati)

class EsploraMemoriaInput(BaseModel):
    categoria_specifica: str = Field(..., description="Il nome della categoria da esplorare. Passa 'TUTTE' per vedere solo l'elenco.")

class EsploraMemoriaTool(BaseTool):
    name: str = "esplora_memoria"
    description: str = "Elenca tutte le categorie esistenti o mostra tutto il contenuto di una singola categoria."
    args_schema: Type[BaseModel] = EsploraMemoriaInput

    def _run(self, categoria_specifica: str) -> str:
        records, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=200, 
            with_payload=True,
            with_vectors=False
        )
        if not records:
            return "Il database è attualmente vuoto."
            
        mappatura = {}
        for r in records:
            cat = r.payload.get("metadata", {}).get("tipo", "generale")
            testo = r.payload.get("page_content", "Senza testo")
            if cat not in mappatura:
                mappatura[cat] = []
            mappatura[cat].append(testo)
            
        if categoria_specifica.upper() != "TUTTE":
            cat_lower = categoria_specifica.lower()
            if cat_lower in mappatura:
                contenuti = "\n".join([f"- {testo}" for testo in mappatura[cat_lower]])
                return f"Contenuto della categoria {cat_lower.upper()}:\n{contenuti}"
            return f"Categoria '{categoria_specifica}' non trovata."
            
        elenco_categorie = "\n".join([f"- {cat.upper()} ({len(elementi)} record)" for cat, elementi in mappatura.items()])
        return f"Categorie attive in memoria:\n{elenco_categorie}"

class EliminaConoscenzaInput(BaseModel):
    vecchia_informazione: str = Field(..., description="La descrizione esatta dell'informazione obsoleta da cancellare.")

class EliminaConoscenzaTool(BaseTool):
    name: str = "elimina_conoscenza"
    description: str = "Cancella un dato obsoleto dal database prima di salvarne uno nuovo."
    args_schema: Type[BaseModel] = EliminaConoscenzaInput

    def _run(self, vecchia_informazione: str) -> str:
        vettore_query = embeddings.embed_query(vecchia_informazione)
        risultati = qdrant_client.search(
            collection_name=collection_name,
            query_vector=vettore_query,
            limit=1
        )
        if risultati:
            id_da_cancellare = risultati[0].id
            testo_cancellato = risultati[0].payload.get("page_content", "Dato senza testo")
            qdrant_client.delete(collection_name=collection_name, points_selector=[id_da_cancellare])
            return f"Memoria aggiornata. Vecchio dato eliminato: '{testo_cancellato}'."
        return "Nessuna informazione pregressa trovata per l'eliminazione."


# ==========================================
# 3. INTERFACCIA UTENTE WEB (STREAMLIT)
# ==========================================
st.title("🤖 Digital PM Colleague - Agile Orchestrator")

tab1, tab2 = st.tabs(["🧠 Memoria & Addestramento", "🚀 Pianificazione Progetto"])

# --- TAB 1: GESTIONE MEMORIA ---
with tab1:
    st.header("Addestra e Interroga il Sistema")
    st.markdown("""
    * **Esplora:** *"Mostrami tutte le categorie che hai in memoria"*
    * **Insegna:** *"Crea la categoria 'baseline_activity' e inserisci questa regola..."*
    * **Interroga:** *"Chi è il referente per l'integrazione Oracle?"*
    * **Correggi:** *"Ti sbagli, il referente non è Mario, è Luigi."*
    """)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    user_input = st.chat_input("Esplora le categorie, scrivi un comando o una regola...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        agente_architetto = Agent(
            role='Knowledge Architect & Memory Manager',
            goal='Gestire la memoria del team: esplorare categorie, rispondere a domande, salvare regole in categorie dinamiche e cancellare dati obsoleti.',
            backstory='Sei il cervello operativo. Se l\'utente esplora, usi il tool di esplorazione. Se fa una domanda mirata, usi la ricerca semantica. Se insegna, salvi (creando categorie se richiesto). Se corregge, elimini il dato obsoleto e salvi il nuovo.',
            tools=[SalvaConoscenzaTool(), RicercaConoscenzaTool(), EsploraMemoriaTool(), EliminaConoscenzaTool()],
            llm=agente_llm,
            verbose=True
        )
        
        task_addestramento = Task(
            description=f'''Analizza: "{user_input}".
            1. Se l'utente vuole ESPLORARE (es. "che categorie hai?"): Usa il tool Esplora Memoria.
            2. Se l'utente fa una DOMANDA MIRATA (es. "chi è il referente per X?"): Usa il tool Ricerca Base di Conoscenza.
            3. Se l'utente fornisce un NUOVO DATO: Usa il tool Salva Conoscenza.
            4. Se è una CORREZIONE: Usa Elimina Conoscenza, poi Salva Conoscenza.
            Fornisci una risposta discorsiva sulle azioni intraprese o sui dati trovati.''',
            expected_output='Risposta che elenca le categorie, mostra i dati trovati o conferma l\'aggiornamento.',
            agent=agente_architetto
        )
        
        with st.spinner("Accesso alla memoria neurale..."):
            risposta = Crew(agents=[agente_architetto], tasks=[task_addestramento]).kickoff()
            st.session_state.chat_history.append({"role": "assistant", "content": risposta.raw})
            st.chat_message("assistant").write(risposta.raw)

# --- TAB 2: PIANIFICAZIONE PROGETTO AGILE ---
with tab2:
    st.header("Generatore WBS Agile Ibrido")
    
    with st.form("project_form"):
        col1, col2 = st.columns(2)
        nome_progetto = col1.text_input("Nome Progetto (es. Migrazione CRM)")
        timeline = col1.text_input("Timeline Stimata (es. 3 mesi)")
        fte = col2.number_input("Risorse Disponibili (FTE)", min_value=1.0, value=3.0, step=0.5)
        
        impatti = st.text_area("Elenco Impatti Tecnici (es. Modifica API Pagamento, Dismissione DB vecchio)")
        submit_btn = st.form_submit_button("Analizza e Genera Piano")
        
    if submit_btn and nome_progetto and impatti:
        st.info("Risveglio agenti e analisi database in corso...")
        
        analista = Agent(
            role='System & Risk Analyst',
            goal='Analizzare gli impatti e trovare referenti e rischi storici dal DB.',
            backstory='Analista tecnico. Usi il tool di ricerca DB per mappare dipendenze organizzative e criticità passate legate ai sistemi impattati.',
            tools=[RicercaConoscenzaTool()],
            llm=agente_llm
        )
        
        planner = Agent(
            role='Agile Delivery Manager',
            goal='Creare una WBS (Epiche e Sprint) coerente con le regole aziendali estratte dall\'analista.',
            backstory='Agile Coach. Strutturi il piano bilanciando FTE, timeline e rischi tecnici forniti.',
            llm=agente_llm
        )
        
        scribe = Agent(
            role='Jira Scribe & Stakeholder Communicator',
            goal='Redigere ticket Jira e comunicazioni.',
            backstory='Traduttore tecnico. Formatti in ticket Jira "As a... I want..." e scrivi email manageriali ai referenti.',
            llm=agente_llm
        )
        
        t1 = Task(
            description=f'Progetto: {nome_progetto}. Impatti: {impatti}. Cerca nel DB chi sono i referenti di questi sistemi e se ci sono rischi storici.',
            expected_output='Elenco referenti e lista rischi tecnici pregressi.',
            agent=analista
        )
        
        t2 = Task(
            description=f'Timeline: {timeline}, Risorse: {fte} FTE. Usa i dati dell\'analista. Crea una WBS Agile suddivisa in Epiche logiche e Sprint.',
            expected_output='Piano di delivery Agile strutturato e bilanciato.',
            agent=planner
        )
        
        t3 = Task(
            description='Genera 3 esempi di User Story in formato Jira con Acceptance Criteria in base alla WBS. Scrivi poi una bozza di email ai referenti individuati.',
            expected_output='Testi pronti per Jira e bozza email manageriale.',
            agent=scribe
        )
        
        with st.spinner("Pianificazione e stesura in corso (potrebbe richiedere un minuto)..."):
            crew_progetto = Crew(agents=[analista, planner, scribe], tasks=[t1, t2, t3], process=Process.sequential)
            risultato_finale = crew_progetto.kickoff()
            
        st.success("Analisi e Pianificazione Completata!")
        st.markdown("### Output Operativo del Team")
        st.markdown(risultato_finale.raw)
