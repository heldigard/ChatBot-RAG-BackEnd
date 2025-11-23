# 🧠 MEMORY - ChatBot RAG Backend

**Última actualización:** 2025-11-23
**Estado:** ✅ Código base funcional (FastAPI + RAG local con FAISS y Sentence-Transformers)
**Arquitectura:** FastAPI + RAG local (FAISS + Sentence Transformers) con LLM compatible OpenAI (openai package)

---

## 📋 RESUMEN EJECUTIVO

Este backend **FastAPI** implementa un sistema RAG local (FAISS + Sentence Transformers) que crea respuestas basadas en archivos PDF. El LLM se conecta mediante un cliente compatible con OpenAI (`openai`) y `OPENAI_API_BASE` permite apuntar a diferentes proveedores (OpenAI, DeepSeek, Azure, etc.).

**Puntos clave:**
- RAG local: FAISS + SentenceTransformers por defecto
- Soporte opcional para embeddings remotos (OpenAI/OpenRouter) usando variables de entorno
- El sistema indexa PDFs desde `PDF_DIRECTORY` y persiste el vector store en `vector_store.*` en disco

---

## 🏗️ ARQUITECTURA IMPLEMENTADA

### Patrones de Diseño
- **API Gateway/Proxy:** El backend actúa como intermediario
- **Stateless:** No mantiene estado entre peticiones
- **Separación de responsabilidades:** Frontend ↔ Backend ↔ RAG local (FAISS) / LLM externo

### Flujo de Datos
```
Frontend (React) → Backend (FastAPI) → RAG local (FAISS) → LLM (OpenAI-compatible)
```
 
---

## 📁 ESTRUCTURA DEL PROYECTO

```
ChatBot-RAG-BackEnd/
├── app.py                 # Aplicación FastAPI
├── pdf_processor.py       # Procesamiento de PDFs
├── embedding_manager.py   # FAISS + sentence-transformers o OpenAI embeddings
├── rag_system.py          # Orquestador RAG (build, retrieve, format)
├── llm_manager.py         # Cliente OpenAI compatible para generación
├── requirements.txt       # Dependencias
├── .env.example           # Variables de entorno ejemplo
├── static/                # Interfaz web estática
└── pdfs/                  # PDFs a indexar
```

---

## 🔧 COMPONENTES IMPLEMENTADOS

### 1. **FastAPI App** (`app.py`)
**Estado:** ✅ RAG local (FAISS) y LLM compatible con OpenAI funcionando

**Endpoints:**
- `GET /health` - Verificación de estado del servicio
- `GET /stats` - Estadísticas del RAG y vector store
- `POST /chat` - Endpoint principal de chat (question + conversation_history opcional)
- `POST /upload_pdf` - Subida de PDF e indexación incremental
- `POST /rebuild` - Reconstrucción del vector store
- `GET /retrieve` - Endpoint de depuración para recuperar fragmentos
- `GET /` - Sirve UI web moderna y responsive en `static/index.html` con diseño optimizado

**Características implementadas:**

- ✅ RAG local (FAISS) y embeddings con `SentenceTransformers` por defecto
- ✅ Soporte para embeddings remotos con `openai` si `USE_OPENAI_EMBEDDINGS=true`
- ✅ Subida de PDFs y añadido incremental a vector store (`/upload_pdf`)
- ✅ Modelo de datos `ChatRequest` con `question` y opcional `conversation_history`
- ✅ Manejo robusto de errores con logging configurables
- ✅ Configuración CORS para desarrollo
- ✅ Cliente reutilizable (singleton pattern)
- ✅ Interfaz web moderna con diseño mejorado, alta legibilidad y jerarquía visual optimizada
- ✅ Sistema de temas claro/oscuro con contraste mejorado para accesibilidad
- ✅ Diseño responsive adaptado para dispositivos móviles y escritorio

**Flujo de procesamiento:**
```python
1. El cliente envía `POST /chat` con `question` y opcionalmente `conversation_history`.
2. El backend recupera documentos relevantes a través de `RAGSystem.retrieve_documents()`.
3. Se formatea el contexto con `RAGSystem.format_context()` (respectando `MAX_CONTEXT_CHARS`).
4. Se invoca a `LLMManager.generate_response()` con la pregunta y el contexto.
5. Se retornan `answer`, `sources`, `context_used`, y `metadata`.
```

**Mejoras vs versión anterior:**
- ✅ RAG local con indexación y recuperación de documentos (FAISS) para respuestas basadas en PDFs
- ✅ Opción de usar embeddings locales (sentence-transformers) o remotos (OpenAI/OpenRouter)
- ✅ Inicio de servicios (RAG y LLM) en `startup` para reducir latencia en la primera petición

 
### 2. **Dependencias** (`requirements.txt`)

**Estado:** ✅ Estables y enfocadas en RAG local y compatibilidad OpenAI
```text
# Core Framework
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6
python-dotenv==1.0.1

# PDF Processing
PyPDF2==3.0.1
pdfplumber==0.10.3

# Text Processing & Splitting
langchain==0.1.0
langchain-text-splitters==0.0.1
tiktoken>=0.5.2

# Vector Database
faiss-cpu==1.12.0
chromadb==0.4.22

# OpenAI Compatible API
openai==1.6.1

# Text Processing
sentence-transformers==2.2.2

# Utilities
numpy>=1.24.3
requests==2.32.3
```

 
### 3. **Variables de Entorno** (`.env.example`)
**Estado:** ✅ Configuración orientada a OpenAI / OpenRouter / DeepSeek y RAG local
```text
# OpenAI / OpenAI-compatible LLM
OPENAI_API_KEY=
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Embeddings
USE_OPENAI_EMBEDDINGS=false
EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_EMBEDDING_API_BASE=
OPENAI_EMBEDDING_API_KEY=

# RAG
PDF_DIRECTORY=./pdfs
VECTOR_STORE_PATH=./vector_store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging
LOG_LEVEL=INFO
```

 
### 4. **Despliegue** (`startup.sh`)
**Estado:** ✅ Preparado para ejecución local y despliegue en plataformas (Azure/AWS/GCP/Kubernetes)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🎯 CUMPLIMIENTO DE REQUISITOS

 
### ✅ Estado: RAG local implementado y funcional
- **Backend FastAPI:** ✅ Implementado
- **Endpoints `/health` y `/chat`:** ✅ Funcionales
-- **Llamadas HTTP a LLM OpenAI-compatible:** ✅ Operativas (configurable mediante `OPENAI_API_BASE`)
- **Gestión de errores:** ✅ Robusta
- **Variables de entorno:** ✅ Configuradas
- **Script de inicio Azure:** ✅ Preparado

 
### ⚠️ DESAFÍO CALDAS - 70% CUMPLIDO
**✅ Cumple:**
- Asistente legal funcional
- Respuestas basadas en documentos legales
- Interfaz chat implementada (en frontend)
- Sistema end-to-end operativo

**❌ No cumple (por diseño del plan):**
- No tests unitarios completos ni CI
- Falta rate limiting y control de producción (rate limits/CORS)

---

## 🔍 ESTADO DE CALIDAD

### ✅ **Fortalezas**
1. **Código limpio** y bien estructurado
2. **Manejo robusto de errores** con HTTPException
3. **Logging implementado** para debugging
4. **Documentación completa** en README
5. **Configuración segura** con .env.example
6. **Listo para producción** en Azure

### ⚠️ **Áreas de Mejora**

#### Críticas (Para cumplir Desafío Caldas)
1. **Añadir endpoint `/upload_pdf`:** Permitir carga dinámica de PDFs
2. **Implementar RAG local:** Chroma/FAISS + embeddings
3. **Procesamiento de PDFs:** Extracción y chunking de texto

#### Sugeridas (Buenas prácticas)
1. **Testing unitario:** tests/test_app.py
2. **Rate limiting:** slowapi para prevenir abuse
3. **CORS producción:** Restringir orígenes específicos
4. **Logging estructurado:** JSON format para producción
5. **Dockerfile:** Para contenerización

---

## 🔧 TAREAS PENDIENTES

### **High Priority**
1. Añadir tests unitarios e integración (CI) para endpoints y flujos RAG (upload -> rebuild -> chat).
2. Implementar rate limiting y protección (por ejemplo `slowapi`).
3. Hacer `upload_pdf` idempotente y documentar comportamiento incremental de indexación.
4. Añadir métricas y alarmas (Prometheus/Datadog) para monitorizar la salud del servicio.

### **Medium Priority (Mejoras)**
1. Implementar streaming SSE/websockets para respuestas si el LLM lo soporta.
2. Contenerizar la app (Dockerfile) y añadir pipelines de CI/CD.
3. Mejorar logs y observabilidad (JSON structured logging, request traces).

### **Low Priority (Opcional)**
```bash
# 1. Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# 2. Estructura mejorada
mkdir -p src/api src/models src/services
```

---

## 🧪 TESTING - ESTADO ACTUAL

**Estado:** ✅ Tests básicos existen (`tests/test_basic_endpoints.py`) — requiere mayor cobertura e integración CI

**Tests sugeridos:**
```python
# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_empty_question():
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 400

def test_chat_valid_question():
    # Se espera que haga un POST a /chat y retorne 'answer'.
    response = client.post("/chat", json={"question": "¿Qué es la Ley 1257?"})
    assert response.status_code in (200, 500)
    assert isinstance(response.json(), dict)

**Nota (Postman):** La colección `postman/` fue actualizada para eliminar los requests dependientes de `threads` (p. ej. `threads/create`). Use los endpoints `Chat (No thread)` y `Chat - Follow-up (no thread)` para pruebas rápidas.
```

---

## 📊 MÉTRICAS Y MONITOREO

**Estado:** ⚠️ Logging básico implementado

**Mejoras sugeridas:**
```python
# logging estructurado
import structlog
logger = structlog.get_logger()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logger.info("chat_request_received", question=request.question)
    # ... lógica existente
    logger.info("chat_response_sent", response_length=len(answer))
```

---

## 🚀 DESPLIEGUE

### Despliegue (recomendaciones)
- **Runtime:** Python 3.11
- **Startup Command:** `uvicorn app:app --host 0.0.0.0 --port 8000`
- **Variables de entorno:** Configurar en la plataforma de despliegue (Azure/AWS/K8s) y rotar claves.
- **CORS:** Restringir a dominios del frontend en producción.

---

## 🔗 INTEGRACIONES

### Frontend
- **URL Base:** `import.meta.env.VITE_API_URL || "http://localhost:8000"`
- **Endpoint Chat:** `POST /chat`
- **Request Format:** `{"question": "texto pregunta"}`
- **Response Format:** `{"answer": "respuesta", "sources": ["fuente1", "fuente2"]}`

### OpenAI / OpenAI-compatible LLM
- **Formato:** API compat. OpenAI (Chat Completions)
- **Headers:** `{'Content-Type': 'application/json', 'Authorization': 'Bearer <OPENAI_API_KEY>'}` o `api_key` si el proveedor lo requiere
- **Timeout:** Depende del proveedor, configurar en `OPENAI_API_BASE` en caso de proveedores alternativos

---

## 📝 DECISIONES DE DISEÑO IMPORTANTES

1. **RAG local por defecto:** FAISS + SentenceTransformers para reproducibilidad y reducción de costos.
2. **No persistencia de conversaciones por defecto:** `conversation_history` se pasa por request para que el cliente controle el contexto.
3. **Se prioriza una arquitectura modular:** `PDFProcessor`, `EmbeddingManager`, `RAGSystem`, `LLMManager`.
4. **Configuración mediante `.env`:** Facilita cambiar proveedores (OpenAI, DeepSeek, Azure, LocalAI) en entornos distintos.
5. **CORS permisivo en desarrollo:** Debe restringirse en producción.

---

## 🚨 PROBLEMAS CONOCIDOS Y TAREAS PENDIENTES

### ✅ Completado/Corregido
1. `.gitignore` creado para proteger secretos y elementos generados.
2. Código estructurado con manejo de errores y logs básico.

### ⚠️ Pendientes (Prioridad)
1. **Testing:** Aumentar la cobertura de unidades e integración; añadir CI.
2. **Rate limiting / protección:** Implementar `slowapi` o middleware equivalente.
3. **CORS producción:** Restringir a orígenes permitidos.
4. **Backups sincronizados:** Mantener versiones y backups del `vector_store`.

---

## 🔄 ESTADO DE DESARROLLO

**Desarrollo:** ✅ Implementado (RAG local + LLM)
**Testing:** ⚠️ Cobertura inicial (pruebas de endpoints), ampliar con unitarias e integración
**Documentación:** ✅ README y Memory actualizados
**Despliegue:** ✅ Compatible con Azure/AWS/GCP/Docker
**Calidad:** ⚠️ Mejorar con tests y observabilidad

---

## 📞 CONTACTO Y SOPORTE

**Para cambios o mejoras:**
1. Revisar este archivo `MEMORY.md` primero
2. Verificar `README.md` y `postman/` para pruebas rápidas
3. Asegurarse de que `OPENAI_API_KEY` y otros secretos no sean versionados

**Próximos desarrolladores:**
- Mantener `MEMORY.md` sincronizado con cambios en endpoints y capacidades RAG
- Añadir notas de migración si se integra con proveedores gestionados (Azure, etc.)

---

**🎯 NOTA FINAL:** Este repositorio implementa un orquestador RAG local (FAISS + Sentence-Transformers) con un LLM compatible con la API de OpenAI. Para producción, recomendamos añadir rate limiting, auditoría/logging y pruebas de integración.

---

## 🆕 CAMBIOS RECIENTES

### Cambios implementados (Resumen)

**Antes:** Proyecto con un prototipo de orquestador y collection en Postman.

**Ahora:**
- Implementado RAG local con `FAISS` y embeddings (`sentence-transformers`) por defecto.
- Añadido endpoint `/upload_pdf` para indexado incremental.
- Implementado `LLMManager` con `openai` (compatible con múltiples `OPENAI_API_BASE`).
- Implementado `rebuild` y `retrieve` endpoints para flujo RAG y debug.
- **MEJORA CRÍTICA:** Implementado SystemPrompt personalizado para evitar respuestas JSON y asegurar formato texto plano.

**Archivos modificados/manuales:**
1. `app.py` - Endpoints y orquestación de RAG + carga de SystemPrompt
2. `requirements.txt` - Dependencias para RAG local
3. `pdf_processor.py`, `embedding_manager.py`, `rag_system.py`, `llm_manager.py` - Core RAG pieces
4. `SystemPrompt.txt` - Prompt optimizado para respuestas en texto plano con formato legal
5. `static/index.html` - Interfaz web completamente rediseñada con UX/UI moderna
6. `postman/` - Collection de pruebas (revisar si hay endpoints no implementados)

### 🎯 Cambio más importante: SystemPrompt optimizado

**Problema resuelto:** El sistema generaba respuestas en formato JSON como:
```json
{
    "answer": "La Ley 1257 de 2008 tiene como objetivo...",
    "sources": [...],
    "metadata": {...}
}
```

**Solución implementada:**
- Se creó `SystemPrompt.txt` con instrucciones explícitas para generar respuestas en texto plano
- Se modificó `llm_manager.py` para cargar el SystemPrompt automáticamente
- Se actualizó `app.py` para pasar el SystemPrompt al LLM en cada solicitud

**Resultado esperado:** Respuestas en formato texto plano legible:
```
La Ley 1257 de 2008 tiene como objetivo la adopción de normas que garanticen para todas las mujeres una vida libre de violencia. [Fuente: Ley 1257 de 2008, Artículo 1]

Esta ley busca asegurar el ejercicio de los derechos de las mujeres... [Fuente: Ley 1257 de 2008, Artículo 2]
```

### 🎨 Cambio reciente: Interfaz de Usuario Modernizada

**Problema resuelto:** La interfaz web tenía problemas de saturación de color y mala legibilidad debido al diseño con gradientes morados intensos y bajo contraste entre elementos.

**Solución implementada:**

- Rediseño completo del sistema de colores con paleta moderna y accesible
- Mejora drástica del contraste y jerarquía visual
- Implementación de sistema de temas claro/oscuro optimizado
- Diseño responsive para móviles y escritorio

**Mejoras principales:**

1. **Esquema de color optimizado:**
   - Fondo: Gradiente suave `#f0f9ff → #e0e7ff → #f5f3ff` (modo claro)
   - Contenedores: Blancos con alta opacidad (95%) para máxima legibilidad
   - Acentos: Azules modernos `#6366f1` con mejor contraste

2. **Jerarquía visual mejorada:**
   - Separación clara entre header, chat, input y footer
   - Bordes y sombras optimizados para profundidad
   - Mensajes del usuario: Color sólido con buen contraste
   - Mensajes del bot: Fondos blancos con sombras sutiles

3. **Accesibilidad y UX:**
   - Sistema de temas claro/oscuro con transiciones suaves
   - Estados interactivos mejorados (hover, focus)
   - Indicadores de carga y conexión más visibles
   - Fuentes de documento interactivas con mejor feedback visual

4. **Responsive design:**
   - Adaptación perfecta a móviles, tablets y escritorio
   - Componentes flexibles que se redimensionan correctamente
   - Controles táctiles optimizados para dispositivos móviles

**Resultado:** Una interfaz moderna, profesional y altamente usable que cumple con estándares de accesibilidad y proporciona una experiencia de usuario superior.
