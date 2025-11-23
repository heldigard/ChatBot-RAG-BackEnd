Write-Output "Removing .env and vector_store files from git tracking..."
git rm --cached .env -ErrorAction SilentlyContinue
git rm --cached vector_store.faiss -ErrorAction SilentlyContinue
git rm --cached vector_store.pkl -ErrorAction SilentlyContinue
git commit -m "Remove sensitive and binary files from repository (.env, vector_store.*)" -ErrorAction SilentlyContinue
Write-Output "Remember to rotate any exposed keys and update your .env locally."
