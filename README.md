# ChatBot RAG Universal

Un chatbot con sistema de Retrieval-Augmented Generation (RAG) que utiliza documentos PDF como base de conocimiento y es compatible con cualquier API que siga el protocolo de OpenAI.

## Características

- Base de conocimiento a partir de documentos PDF
- Compatible con cualquier API que siga el protocolo de OpenAI
- Interfaz de chat web intuitiva
- Sistema de embeddings local o con OpenAI
- Recuperación de documentos relevantes con fuentes
- Historial de conversación

## Arquitectura

El sistema está compuesto por los siguientes módulos:

- **PDFProcessor**: Extrae y procesa texto de archivos PDF
- **EmbeddingManager**: Gestiona embeddings y búsqueda vectorial
- **RAGSystem**: Orquesta la recuperación de documentos
- **LLMManager**: Interactúa con APIs compatibles con OpenAI
- **FastAPI Backend**: Expone los endpoints de la API

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd ChatBot-RAG-BackEnd
```

2. Crear un entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Configuración

1. Copiar el archivo de entorno de ejemplo:
```bash
cp .env.example .env
```

2. Configurar las variables de entorno en el archivo `.env`:

```env
# Variables de entorno para la API compatible con OpenAI
OPENAI_API_KEY=tu_api_key_aqui
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Variables de entorno para la configuración RAG
USE_OPENAI_EMBEDDINGS=false
EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_EMBEDDING_API_BASE=
OPENAI_EMBEDDING_API_KEY=
PDF_DIRECTORY=./pdfs
VECTOR_STORE_PATH=./vector_store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Configuración de API

Para usar con OpenAI:
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

Para usar con otra API compatible con OpenAI (ej. Azure, LocalAI, etc.):
```env
OPENAI_API_KEY=tu_api_key_personalizada
OPENAI_API_BASE=https://tu-endpoint.com/v1
OPENAI_MODEL=tu_modelo_disponible
```

### Configuración de Embeddings

Para usar embeddings locales (recomendado para desarrollo):
```env
USE_OPENAI_EMBEDDINGS=false
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Para usar embeddings de OpenAI:
```env
USE_OPENAI_EMBEDDINGS=true
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

Para usar una URL diferente para los embeddings:
```env
USE_OPENAI_EMBEDDINGS=true
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_EMBEDDING_API_BASE=https://tu-api-de-embeddings.com/v1
```

Para usar diferentes API keys para LLM y embeddings:
```env
OPENAI_API_KEY=tu_api_key_para_llm
OPENAI_EMBEDDING_API_KEY=tu_api_key_para_embeddings
```

### Configuración con DeepSeek y OpenRouter

Ejemplo para usar DeepSeek como LLM y OpenRouter para embeddings:
```env
# Configuración de DeepSeek (LLM)
OPENAI_API_KEY=tu_api_key_de_deepseek
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# Configuración de OpenRouter (Embeddings)
USE_OPENAI_EMBEDDINGS=true
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENAI_EMBEDDING_API_BASE=https://openrouter.ai/api/v1
OPENAI_EMBEDDING_API_KEY=tu_api_key_de_openrouter
```

Nota: Si `OPENAI_EMBEDDING_API_BASE` está vacío, se usará `OPENAI_API_BASE` para los embeddings.

## Uso

1. Colocar los archivos PDF en el directorio `pdfs/`:
```bash
mkdir -p pdfs
cp tus-documentos*.pdf pdfs/
```

2. Iniciar el servidor:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

3. Acceder a la interfaz de chat en:
```
http://localhost:8000
```

## API Endpoints

### Chat
- **URL**: `/chat`
- **Método**: POST
- **Descripción**: Envía una pregunta al sistema RAG

**Cuerpo de la solicitud**:
```json
{
  "question": "¿Cuál es el propósito de la Constitución de Colombia?",
  "conversation_history": []
}
```

**Respuesta**:
```json
{
  "answer": "La Constitución de Colombia establece...",
  "sources": [
    {
      "filename": "COLOMBIA-Constitucion.pdf",
      "page": 1,
      "score": 0.85
    }
  ],
  "context_used": true,
  "retrieved_docs_count": 3,
  "metadata": {}
}
```

### Estadísticas
- **URL**: `/stats`
- **Método**: GET
- **Descripción**: Obtiene estadísticas del sistema RAG

**Respuesta**:
```json
{
  "total_documents": 3,
  "total_chunks": 245,
  "embedding_model": "all-MiniLM-L6-v2",
  "vector_store_path": "./vector_store"
}
```

### Reconstruir Vector Store
- **URL**: `/rebuild`
- **Método**: POST
- **Descripción**: Reconstruye el vector store con los PDFs actuales

**Cuerpo (JSON)**:
```json
{
  "force": false
}
```

### Upload PDF (nueva)

## Vector Store: archivos, reinicio y buenas prácticas

Este proyecto utiliza FAISS para la base de datos vectorial y persiste el índice y los metadatos en disco. Por defecto, la ruta base del vector store es la variable de entorno `VECTOR_STORE_PATH`. Si `VECTOR_STORE_PATH` está configurado a `./vector_store` el sistema creará los archivos:

- `./vector_store.faiss` — archivo binario FAISS con el índice
- `./vector_store.pkl` — archivo `pickle` con los documentos, embeddings y metadatos

Por lo tanto, cuando se agregan, eliminan o modifican PDFs, es importante reconstruir (o eliminar) el vector store para evitar contenido obsoleto o duplicado.

Opciones para mantener el vector store en sincronía con los PDFs:

1. Usar el endpoint `/rebuild` (recomendado cuando haya cambios grandes):

```bash
# Reconstruye completamente los archivos del vector store
curl -X POST -H "Content-Type: application/json" -d '{"force": true}' http://localhost:8000/rebuild
```

1. Subir PDFs individualmente (indexado incremental):

```bash
curl -X POST -F "file=@/path/to/doc.pdf" http://localhost:8000/upload_pdf
```

Nota: `/upload_pdf` añade los fragmentos del PDF al índice existente; no elimina entradas antiguas si reemplazas un PDF con el mismo nombre. Si reemplazas un documento existente o modificas su contenido, **ejecuta `/rebuild`** o **elimina manualmente** los archivos `*.faiss` y `*.pkl` y reinicia el servidor para reconstruir el índice desde cero.

1. Eliminar manualmente los archivos (PowerShell / Bash):

PowerShell (Windows):
```powershell
Remove-Item -Path .\vector_store.faiss -ErrorAction SilentlyContinue
Remove-Item -Path .\vector_store.pkl -ErrorAction SilentlyContinue
```

Bash (Linux / macOS / WSL):
```bash
rm -f ./vector_store.faiss ./vector_store.pkl
```

1. Consideraciones al usar `/upload_pdf`:

- Es útil cuando agregas nuevos PDFs y quieres que estén indexados sin reconstruir todo el índice.
- No es suficiente cuando reemplazas PDFs con cambios en su contenido con el mismo nombre — esto puede dejar duplicados o antiguos fragmentos en el índice.
- Si necesitas controlar duplicados, haz un `rebuild` o borra manualmente los archivos del vector store y reconstruye.

1. Otros consejos prácticos:

- Mantén el directorio `pdfs/` limpio y ordenado — borra versiones obsoletas y evita nombres duplicados.
- Para grandes sets de PDFs (miles), evalúa una estrategia de versión para los archivos y/o el uso de almacenamiento persistente con backup del vector store.
- Versionar `vector_store` en git no es recomendable (archivos binarios), y ya existen scripts en `scripts/` para eliminar la huella del repo (`remove_sensitive.*`).
 

### Retrieve (nueva)
- **URL**: `/retrieve?query=tu_pregunta&k=5`
- **Método**: GET
- **Descripción**: Endpoint de desarrollo para ver los fragmentos recuperados por la consulta (útil para debugging/construcción de prompts).

**Respuesta**:
```json
{
  "query": "¿Qué es ...?",
  "results": [
    { "content": "...", "metadata": {"filename":"..."}, "score": 0.7 },
    ...
  ]
}
```


## Estructura del Proyecto

```
ChatBot-RAG-BackEnd/
├── app.py                 # Aplicación FastAPI principal
├── pdf_processor.py        # Procesamiento de PDFs
├── embedding_manager.py    # Gestión de embeddings
├── rag_system.py         # Sistema RAG
├── llm_manager.py        # Gestión de LLM
├── requirements.txt       # Dependencias
├── .env.example         # Variables de entorno ejemplo
├── pdfs/               # Directorio para PDFs
├── vector_store/        # Directorio o prefijo para vector store (ver notas abajo)
└── static/
    └── index.html       # Interfaz de chat web
```

## Personalización

### Cambiar Modelo de Embeddings

Para usar un modelo diferente de Sentence Transformers:

1. Cambiar la variable `EMBEDDING_MODEL` en `.env`
2. Reconstruir el vector store con el endpoint `/rebuild`

### Ajustar Parámetros de Chunking

Para modificar cómo se dividen los documentos:

1. Ajustar `CHUNK_SIZE` y `CHUNK_OVERLAP` en `.env`
2. Reconstruir el vector store con el endpoint `/rebuild`

## Solución de Problemas

### Error: "No module named 'distutils'"

Instalar setuptools:
```bash
pip install setuptools
```

### Error: "Can't find Rust compiler" al instalar tiktoken

Instalar una versión precompilada:
```bash
pip install tiktoken --upgrade
```

### El servidor no inicia

Verificar que:
1. Todas las dependencias estén instaladas
2. El archivo `.env` esté configurado correctamente
3. Los archivos PDF estén en el directorio `pdfs/`

## Seguridad y manejo de secretos ⚠️

Es importante no subir claves ni secretos al repositorio. Este proyecto usa un archivo `.env` para desarrollo local. Siga estas recomendaciones:

- No incluya claves reales en el repositorio ni en `.env.example`.
- Asegúrese de que `.env` esté incluido en `.gitignore` (por defecto ya lo está en este proyecto).
- Si accidentalmente subió una clave en el repositorio, **rotela** inmediatamente desde el proveedor de la API.
- Para eliminar un `.env` ya comiteado: (a) eliminarlo del repo y (b) rotar las claves: 

```bash
git rm --cached .env
git commit -m "Remove .env from repository"
git push
```

- Considere usar un gestor de secretos (Azure Key Vault, AWS Secrets Manager, HashiCorp Vault, GitHub Secrets) para entornos de producción en vez de `.env`.

## Licencia

Este proyecto está bajo la Licencia MIT.
