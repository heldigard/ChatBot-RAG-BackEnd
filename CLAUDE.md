# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **ChatBot RAG Universal Backend** - a Retrieval-Augmented Generation system that processes PDF documents and answers questions using OpenAI-compatible APIs. The system is designed to work with custom LLM providers (like DeepSeek, OpenRouter) and supports both local (SentenceTransformer) and OpenAI embeddings with FAISS vector indexing.

**Current Status**: ✅ Fully functional RAG system with local embeddings (FAISS + SentenceTransformers) and OpenAI-compatible LLM integration. The system includes legal Colombian documents for testing and supports dynamic PDF upload and indexing.

## Development Commands

### Running the Application
```bash
# Start the development server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Alternative using the startup script
bash startup.sh
```

### Setting Up Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Vector Store Management
```bash
# Rebuild vector store via API
curl -X POST http://localhost:8000/rebuild -H "Content-Type: application/json" -d '{"force": true}'

# Get system statistics
curl http://localhost:8000/stats
```

## Architecture Overview

### Core Components

The system follows a modular RAG architecture:

1. **PDFProcessor** (`pdf_processor.py`) - Extracts text from PDFs using pdfplumber with PyPDF2 fallback, splits documents into chunks using langchain text splitters
2. **EmbeddingManager** (`embedding_manager.py`) - Manages embeddings using either SentenceTransformer (local) or OpenAI APIs, with FAISS vector indexing
3. **RAGSystem** (`rag_system.py`) - Orchestrates document retrieval and context formatting, manages vector store lifecycle
4. **LLMManager** (`llm_manager.py`) - Handles interactions with OpenAI-compatible APIs for response generation
5. **FastAPI App** (`app.py`) - Main application with CORS, health checks, chat endpoint, and static file serving

### Data Flow
```
PDFs → PDFProcessor → Text Chunks → EmbeddingManager → FAISS Index
User Query → RAGSystem → Document Retrieval → Context Formatting → LLMManager → Response
```

### Configuration Architecture

The system is highly configurable through environment variables:

- **LLM Configuration**: Supports any OpenAI-compatible API (DeepSeek, OpenRouter, etc.)
- **Embedding Configuration**: Choice between local SentenceTransformer or OpenAI embeddings
- **RAG Parameters**: Configurable chunk size, overlap, and retrieval parameters
- **File Paths**: Configurable PDF directory and vector store location

## Key Files and Their Roles

### `app.py` - Main FastAPI Application
- **Endpoints**: `/health`, `/chat`, `/stats`, `/rebuild`, `/upload_pdf`, `/retrieve`
- **Singleton Pattern**: Manages RAGSystem and LLMManager instances for performance
- **CORS Configuration**: Development-friendly (restrict in production)
- **Static File Serving**: Serves modern HTML interface from `/static`
- **SystemPrompt Integration**: Loads custom system prompt for optimal response formatting
- **Conversation History**: Supports optional conversation context in chat requests
- **Enhanced UI/UX**: Modern responsive interface with improved accessibility and visual hierarchy

### `SystemPrompt.txt` - Custom System Prompt
- **Response Format**: Enforces plain text responses (no JSON)
- **Legal Context**: Optimized for Colombian legal document analysis
- **Citation Requirements**: Ensures proper source attribution
- **Fallback Handling**: Graceful degradation when prompt file unavailable

### `rag_system.py` - RAG Orchestration
- Coordinates PDF processing and embedding management
- Handles vector store persistence (FAISS + pickle)
- Implements document retrieval and context formatting
- Provides system statistics and rebuild functionality

### `embedding_manager.py` - Vector Operations
- Dual embedding support: SentenceTransformer (local) vs OpenAI (API)
- FAISS index management for similarity search
- Vector store persistence with metadata
- Batch processing for API efficiency

### `llm_manager.py` - LLM Integration
- **OpenAI-Compatible Client**: Works with OpenAI, DeepSeek, OpenRouter, etc.
- **Custom SystemPrompt Loading**: Automatically loads and integrates system prompts
- **Conversation History**: Maintains context across multiple interactions
- **Error Handling**: Comprehensive error management with fallback prompts
- **Token Usage Tracking**: Monitors API consumption for cost optimization

### `postman/` - API Testing Collection
- **Updated Collections**: Current endpoints without deprecated thread dependencies
- **Environment Setup**: Preconfigured variables for local testing
- **Test Scenarios**: Health checks, chat interactions, PDF uploads
- **Debug Endpoints**: Include `/retrieve` for RAG system testing

### `pdf_processor.py` - Document Processing
- Dual PDF extraction: pdfplumber (primary) with PyPDF2 fallback
- Recursive character text splitting with configurable parameters
- Metadata enrichment (filename, chunk_id, page numbers)
- Batch processing of multiple PDFs

### `static/index.html` - Modern Web Interface

- **Enhanced Visual Design**: Modern gradient backgrounds with improved color contrast
- **Responsive Layout**: Fully adaptive design for mobile, tablet, and desktop
- **Accessibility Features**: High contrast mode, clear typography, and keyboard navigation
- **Interactive Elements**: Hover states, smooth transitions, and visual feedback
- **Theme System**: Light/dark mode toggle with optimized color schemes
- **Component Architecture**: Modular CSS with custom properties for maintainability
- **Connection Management**: Real-time status indicators and error handling
- **Message Display**: Enhanced chat interface with source attribution and metadata

## Configuration

### Environment Variables (.env.example)
```bash
# LLM Configuration (DeepSeek/OpenRouter/OpenAI)
OPENAI_API_KEY="your-api-key"
OPENAI_API_BASE="https://api.deepseek.com/v1"  # Custom API endpoint
OPENAI_MODEL="deepseek-chat"

# Embedding Configuration
USE_OPENAI_EMBEDDINGS=false  # false = local SentenceTransformer (recommended)
EMBEDDING_MODEL="all-MiniLM-L6-v2"  # Local model: fast, free, private
OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"  # If using OpenAI embeddings
OPENAI_EMBEDDING_API_BASE=""  # Optional different endpoint for embeddings
OPENAI_EMBEDDING_API_KEY=""  # Optional different API key for embeddings

# RAG Configuration
PDF_DIRECTORY="./pdfs"  # Directory containing PDF documents
VECTOR_STORE_PATH="./vector_store"  # Base path for vector store files
CHUNK_SIZE=1000  # Characters per document chunk (800-1200 for legal texts)
CHUNK_OVERLAP=200  # Overlapping characters between chunks (15-20% of chunk_size)

# Logging Configuration
LOG_LEVEL=INFO  # DEBUG for detailed troubleshooting
```

### Vector Store Structure
- `./vector_store.faiss` - FAISS index file for fast similarity search
- `./vector_store.pkl` - Serialized documents, embeddings, and metadata

### RAG Parameters Explained
- **CHUNK_SIZE**: Controls document fragment size (1000 = balance between context and precision)
- **CHUNK_OVERLAP**: Ensures context continuity across chunks (200 = 20% overlap)
- **Embedding Choice**: Local SentenceTransformer (free, private) vs OpenAI (more accurate, costs money)

## Testing the System

### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

### Chat Example
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué protege la Ley 1257 de 2008?",
    "conversation_history": []
  }'
```

### Upload PDF Example
```bash
curl -X POST http://localhost:8000/upload_pdf \
  -H "Content-Type: multipart/form-data" \
  -F "file=@new_document.pdf"
```

### System Statistics
```bash
curl http://localhost:8000/stats
# Returns document count, model info, paths, and RAG configuration
```

### Debug Document Retrieval
```bash
curl "http://localhost:8000/retrieve?query=derechos%20fundamentales&k=3"
# Returns top 3 most relevant document chunks with similarity scores
```

## Common Development Patterns

### Adding New Documents
1. Place PDFs in the `pdfs/` directory
2. Call `/rebuild` endpoint or restart the application
3. Vector store will automatically update with new documents

### Switching Embedding Models
1. Update `USE_OPENAI_EMBEDDINGS` and related variables in `.env`
2. Call `/rebuild` endpoint with `{"force": true}`
3. System will reprocess all documents with new embedding model

### Debugging Embedding Issues
- Check vector store files exist: `vector_store.faiss` and `vector_store.pkl`
- Verify PDF directory contains valid PDF files
- Monitor logs for embedding processing errors
- Use `/stats` endpoint to verify document count

## Production Considerations

- **CORS**: Currently allows all origins - restrict to specific domains in production
- **API Keys**: Use environment variables, never commit to repository
- **Error Handling**: Comprehensive error handling with HTTP status codes
- **Logging**: Basic logging configured - enhance for production monitoring
- **File Paths**: Use absolute paths in production environments
- **Rate Limiting**: Consider implementing rate limiting for production usage
- **Monitoring**: Add health checks and metrics for production monitoring

## Document Store

The system includes legal documents for testing:
- Colombian Constitution (`COLOMBIA-Constitucion.pdf`) - Fundamental rights and state organization
- Ley 769 de 2002 (`LEY 769 DE 2002.pdf`) - Traffic regulations and vehicle codes
- Ley 1257 de 2008 (`Ley_1257_de_2008.pdf`) - Violence against women prevention and protection

## Response Format and SystemPrompt

The system uses a custom `SystemPrompt.txt` to ensure responses are:
- **Plain Text Format**: No JSON structures or markdown formatting
- **Legal Context**: Optimized for Colombian legal document analysis
- **Source Citation**: Automatic citation of documents and articles
- **Fallback Behavior**: Graceful handling when information is not found

### Example Response Format
```
La Ley 1257 de 2008 tiene como objetivo la adopción de normas que garanticen para todas las mujeres una vida libre de violencia.

Esta ley busca asegurar el ejercicio de los derechos de las mujeres reconocidos en el ordenamiento jurídico. [Fuente: Ley 1257 de 2008, Artículo 1]

Define la violencia contra la mujer como cualquier acción u omisión que cause muerte, daño o sufrimiento. [Fuente: Ley 1257 de 2008, Artículo 2]
```

## Architecture Strengths

### Why FAISS for This Project
- **Performance**: Ultra-fast similarity search for <1000 document chunks
- **Simplicity**: Only 2 files to manage (.faiss + .pkl)
- **Resource Efficiency**: Lower memory footprint compared to alternatives
- **Deployment**: Easy to copy and deploy across environments

### Local vs Remote Embeddings
- **Local (SentenceTransformer)**: Recommended for this project - free, private, fast
- **Remote (OpenAI)**: More accurate but adds cost and dependency
- **Hybrid Approach**: System supports switching between them via environment variables

## Testing and Development

### Unit Tests
```bash
# Run basic endpoint tests
python -m pytest tests/test_basic_endpoints.py -v
```

### Postman Collection
- Use `postman/ChatBot_RAG_Backend.postman_collection.json` for comprehensive testing
- Includes all current endpoints with sample requests
- Environment setup with local testing variables

### Debug Mode
Set `LOG_LEVEL=DEBUG` in `.env` for detailed troubleshooting information including:
- Document processing steps
- Embedding generation details
- Vector search operations
- LLM API interactions