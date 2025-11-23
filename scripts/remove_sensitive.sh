#!/usr/bin/env bash
# Script para remover archivos sensibles y grandes del repositorio (ejecutar desde la raíz del repo)
set -e
echo "Removing .env and vector_store files from git tracking..."
git rm --cached .env || true
git rm --cached vector_store.faiss || true
git rm --cached vector_store.pkl || true
git commit -m "Remove sensitive and binary files from repository (.env, vector_store.*)" || true
echo "Remember to rotate any exposed keys and update your .env locally."
