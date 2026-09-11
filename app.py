import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

# ==========================================
# 1. SETUP E CONNESSIONI CLOUD
# ==========================================
st.set_page_config(page_title="AI PM Digital Colleague", page_icon="🚀", layout="wide")

# Recupero chiavi in modo sicuro dai Secrets di Streamlit Cloud
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))

if not GOOGLE_API_KEY or not QDRANT_URL:
    st.error("⚠️ Chiavi API mancanti. Configura i Secrets su Streamlit Cloud.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Inizializza Motore Logico (LLM) ed Embeddings (Vettorializzazione)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Connessione al Database Vettoriale Permanente (Qdrant)
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
vectorstore = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embeddings)


# ==========================================
# 2. DEFINIZIONE TOOLS DI MEMORIA RAG
# ==========================================
@tool("Salva Conoscenza Aziendale")
def tool_salva_conoscenza(testo: str, categoria: str) -> str:
    """Salva una nuova regola, referente o metodo. Categorie ammesse: 'regola_metodologica', 'mappa_stakeholder', 'storico_progetti'."""
    doc = Document(page_content=testo, metadata={"tipo": categoria})
    vectorstore.add_documents([doc])
    return f"Fatto. Informazione salvata in memoria nella categoria: {categoria}."

@tool("Ricerca Base di Conoscenza")
def tool_ricerca_conoscenza(query: str) -> str:
    """Usa questo tool per rispondere a domande sullo storico o per recuperare regole e referenti."""
    docs = vectorstore.similarity_search(query, k=4)
    if not docs:
        return "Non ho trovato informazioni in memoria a riguardo."
    risultati = [f"[{d.metadata.get('tipo', 'generico').upper()}]: {d.page_content}" for d in docs]
    return "\n\n".join(risultati)

@tool("Elimina e Sovrascrivi Conoscenza")
def tool_elimina_conoscenza(vecchia_informazione: str) -> str:
    """Usa questo tool se l'utente corregge una regola/referente obsoleto. Passa in input la VECCHIA informazione da cercare e distruggere."""
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
    * **Insegna:** *"Le epiche non durano più di 1 mese."*
    * **Interroga:** *"Chi è il referente per l'integrazione Oracle?"*
    * **Correggi:** *"Ti sbagli, il referente non è Mario, è Luigi."*
    """)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    user_input = st.chat_input("Scrivi un comando o una regola...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Agente 1: Architetto della Conoscenza
        agente_architetto = Agent(
            role='Knowledge Architect & Memory Manager',
            goal='Gestire la memoria del team: rispondere a domande, salvare regole e cancellare dati vecchi.',
            backstory='Sei il cervello operativo. Se l\'utente fa una domanda, cerchi nel DB. Se insegna qualcosa, lo salvi. Se corregge, elimini prima il dato obsoleto tramite tool.',
            tools=[tool_salva_conoscenza, tool_ricerca_conoscenza, tool_elimina_conoscenza],
            llm=llm,
            verbose=True
        )
        
        task_addestramento = Task(
            description=f'''Analizza: "{user_input}".
            1. Se DOMANDA: Cerca e rispondi.
            2. Se NUOVO DATO: Salva nel DB.
            3. Se CORREZIONE: Usa PRIMA il tool elimina per distruggere il vecchio dato, POI salva il nuovo.
            Fornisci una risposta discorsiva sulle azioni intraprese.''',
            expected_output='Risposta che conferma il salvataggio, la ricerca o la cancellazione avvenuta nel database.',
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
        
        # Agente 2: Analista
        analista = Agent(
            role='System & Risk Analyst',
            goal='Analizzare gli impatti e trovare referenti e rischi storici dal DB.',
            backstory='Analista tecnico. Usi il tool di ricerca DB per mappare dipendenze organizzative e criticità passate legate ai sistemi impattati.',
            tools=[tool_ricerca_conoscenza],
            llm=llm
        )
        
        # Agente 3: Planner
        planner = Agent(
            role='Agile Delivery Manager',
            goal='Creare una WBS (Epiche e Sprint) coerente con le regole aziendali estratte dall\'analista.',
            backstory='Agile Coach. Strutturi il piano bilanciando FTE, timeline e rischi tecnici forniti.',
            llm=llm
        )
        
        # Agente 4: Scribe
        scribe = Agent(
            role='Jira Scribe & Stakeholder Communicator',
            goal='Redigere ticket Jira e comunicazioni.',
            backstory='Traduttore tecnico. Formatti in ticket Jira "As a... I want..." e scrivi email manageriali ai referenti.',
            llm=llm
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
