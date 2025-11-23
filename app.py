import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import UploadFile, File
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag_system import RAGSystem
from llm_manager import LLMManager

# Load environment variables from a .env file (local development)
load_dotenv()

def load_system_prompt() -> str:
    """Carga el SystemPrompt desde el archivo."""
    try:
        with open('SystemPrompt.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error al cargar SystemPrompt.txt: {e}")
        # Fallback a un prompt básico
        return """
Eres un asistente legal experto. Responde basándote únicamente en la información proporcionada.
Si no encuentras la respuesta en los documentos, indica que no tienes esa información.
"""

# Basic logging (configurable with LOG_LEVEL env var)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="ChatBot RAG Universal - Backend")

# CORS middleware: allow all origins for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables for DeepSeek API (LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Usamos la misma variable para compatibilidad
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
if OPENAI_API_KEY:
    logger.debug("OPENAI_API_KEY is set (masked): %s", f"{OPENAI_API_KEY[:4]}...{OPENAI_API_KEY[-4:]}")
else:
    logger.debug("OPENAI_API_KEY is not set")
logger.debug("OPENAI_API_BASE = %s", OPENAI_API_BASE)
logger.debug("OPENAI_MODEL = %s", OPENAI_MODEL)

# Environment variables for OpenRouter API (Embeddings)
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
logger.debug("EMBEDDING_MODEL = %s", EMBEDDING_MODEL)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_EMBEDDING_API_BASE = os.getenv("OPENAI_EMBEDDING_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY", "")
if OPENAI_EMBEDDING_API_KEY:
    logger.debug("OPENAI_EMBEDDING_API_KEY is set (masked): %s", f"{OPENAI_EMBEDDING_API_KEY[:4]}...{OPENAI_EMBEDDING_API_KEY[-4:]}")
else:
    logger.debug("OPENAI_EMBEDDING_API_KEY is not set")
logger.debug("OPENAI_EMBEDDING_API_BASE = %s", OPENAI_EMBEDDING_API_BASE)
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
try:
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
except Exception:
    MAX_CONTEXT_CHARS = 4000
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
except Exception:
    logger.warning("CHUNK_SIZE environment variable is invalid, using default 1000")
    CHUNK_SIZE = 1000
try:
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
except Exception:
    logger.warning("CHUNK_OVERLAP environment variable is invalid, using default 200")
    CHUNK_OVERLAP = 200

# Global instances
rag_system: Optional[RAGSystem] = None
llm_manager: Optional[LLMManager] = None


def get_rag_system() -> RAGSystem:
    """Initialize and return the RAG system."""
    # Prefer RAG stored in app.state (FastAPI) to avoid global state
    rag_from_state = getattr(app.state, 'rag_system', None)
    if rag_from_state is not None:
        return rag_from_state

    global rag_system

    if rag_system is None:
        try:
            rag_system = RAGSystem(
                pdf_directory=PDF_DIRECTORY,
                vector_store_path=VECTOR_STORE_PATH,
                use_openai_embeddings=USE_OPENAI_EMBEDDINGS,
                openai_api_key=OPENAI_EMBEDDING_API_KEY if USE_OPENAI_EMBEDDINGS else OPENAI_API_KEY,
                openai_api_base=OPENAI_EMBEDDING_API_BASE if USE_OPENAI_EMBEDDINGS else OPENAI_API_BASE,
                openai_embedding_api_base=OPENAI_EMBEDDING_API_BASE,
                openai_embedding_api_key=OPENAI_EMBEDDING_API_KEY,
                embedding_model=EMBEDDING_MODEL,
                openai_embedding_model=OPENAI_EMBEDDING_MODEL,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            logger.info("Sistema RAG inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar sistema RAG: {e}")
            raise

    # store in app.state
    setattr(app.state, 'rag_system', rag_system)
    return rag_system


def get_llm_manager() -> LLMManager:
    """Initialize and return the LLM manager."""
    # Prefer LLM manager stored in app.state to avoid global state
    llm_from_state = getattr(app.state, 'llm_manager', None)
    if llm_from_state is not None:
        return llm_from_state

    global llm_manager

    if llm_manager is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY no está configurado")

        try:
            logger.debug("Inicializando LLM Manager (api_key masked) api_base='%s', model='%s'", OPENAI_API_BASE, OPENAI_MODEL)
            llm_manager = LLMManager(
                api_key=OPENAI_API_KEY,
                api_base=OPENAI_API_BASE,
                model=OPENAI_MODEL
            )
            # Log the exact model actually used by LLMManager (read from the instance to avoid hardcoded names)
            logger.info("LLM Manager inicializado correctamente con modelo: %s", llm_manager.model)
        except Exception as e:
            logger.error(f"Error al inicializar LLM Manager: {e}")
            raise

    return llm_manager
# Pydantic models for incoming requests
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="La pregunta del usuario")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Historial de conversación (opcional)")


class RebuildRequest(BaseModel):
    force: bool = Field(False, description="Forzar reconstrucción completa")


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    """Retorna estadísticas del sistema RAG."""
    try:
        rag = get_rag_system()
        stats = rag.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild_vector_store(request: RebuildRequest):
    """Reconstruye el vector store de documentos.

    Args:
        request: RebuildRequest con opción de forzar reconstrucción

    Returns:
        dict: Estado de la operación
    """
    try:
        rag = get_rag_system()

        if request.force:
            rag.rebuild_vector_store()
            message = "Vector store reconstruido completamente"
        else:
            # Verificar si hay cambios en los PDFs
            # Esta es una implementación simple, se podría mejorar
            rag.rebuild_vector_store()
            message = "Vector store actualizado"

        logger.info(message)
        return {"status": "success", "message": message}
    except Exception as e:
        logger.error(f"Error al reconstruir vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Sube un PDF y lo añade al vector store.

    Returns:
        dict: Estado de la operación y metadatos del archivo
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")

    try:
        pdf_dir = PDF_DIRECTORY
        os.makedirs(pdf_dir, exist_ok=True)
        dest_path = os.path.join(pdf_dir, file.filename)
        with open(dest_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        rag = get_rag_system()
        # Procesar el PDF y añadir al vector store
        chunks = rag.pdf_processor.process_pdf(dest_path)
        if not chunks:
            return {"status": "warning", "message": "PDF subido pero no se extrajo texto"}

        rag.add_documents(chunks)
        return {"status": "success", "filename": file.filename, "chunks_added": len(chunks)}

    except Exception as e:
        logger.exception("Error al subir PDF")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Procesa una pregunta usando el sistema RAG.

    Flujo:
    1. Recupera documentos relevantes
    2. Genera respuesta con LLM usando contexto
    3. Retorna respuesta y fuentes

    Args:
        request: ChatRequest con question y opcionalmente conversation_history

    Returns:
        dict: Contiene answer, sources y metadata
    """
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía."
        )

    try:
        # Inicializar sistemas
        rag = get_rag_system()
        llm = get_llm_manager()

        # Recuperar documentos relevantes
        logger.info(f"Recuperando documentos para: {question}")
        retrieved_docs = rag.retrieve_documents(question, k=5)

        # Formatear contexto (limitamos longitud para evitar exceder límites de token)
        context = rag.format_context(retrieved_docs, max_context_chars=MAX_CONTEXT_CHARS)

        # Extraer información de fuentes
        sources = rag.get_sources_info(retrieved_docs)

        # Cargar el SystemPrompt personalizado
        system_prompt = load_system_prompt()

        # Generar respuesta
        logger.info("Generando respuesta con LLM")
        response = llm.generate_response(
            query=question,
            context=context,
            system_prompt=system_prompt,
            conversation_history=request.conversation_history
        )

        answer = response.get("answer", "No se pudo generar respuesta")
        metadata = response.get("metadata", {})

        logger.info(f"Respuesta generada exitosamente")

        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(retrieved_docs) > 0,
            "retrieved_docs_count": len(retrieved_docs),
            "metadata": metadata
        }

    except ValueError as e:
        logger.error(f"Error de configuración: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error al procesar chat")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la solicitud: {str(e)}"
        )


@app.get("/retrieve")
def retrieve(query: str, k: int = 5):
    """Endpoint de desarrollo para obtener documentos recuperados por el RAG (sin llamar al LLM)."""
    if not query:
        raise HTTPException(status_code=400, detail="Parámetro 'query' es requerido")
    try:
        rag = get_rag_system()
        retrieved_docs = rag.retrieve_documents(query, k=k)
        docs = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in retrieved_docs
        ]
        return {"query": query, "results": docs}
    except Exception as e:
        logger.exception("Error en retrieve")
        raise HTTPException(status_code=500, detail=str(e))


# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    """Sirve la página principal del chatbot."""
    return FileResponse('static/index.html')


@app.on_event("startup")
async def startup_event():
    """Inicializa el sistema RAG y el LLM al iniciar el servidor para evitar latencias en la primera solicitud."""
    logger.info("Iniciando startup: inicializando RAG y LLM")
    try:
        # Inicializar sin lanzar excepciones para evitar fallo en el arranque
        get_rag_system()
    except Exception as e:
        logger.warning("No se pudo inicializar RAG en startup: %s", e)
    try:
        get_llm_manager()
    except Exception as e:
        logger.warning("No se pudo inicializar LLM en startup: %s", e)
