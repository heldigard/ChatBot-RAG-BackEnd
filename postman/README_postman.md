# Postman - Hackaton Talento Tech - Backend

This folder contains a Postman Collection and an Environment to test the backend locally.

Files:
- `Hackaton_Talento_Tech_Backend.postman_collection.json` - Postman collection with tests
- `Hackaton_Talento_Tech_Backend.postman_environment.json` - Postman environment with required variables

How to import and run tests in Postman:
1. Open Postman, click Import → File → select `Hackaton_Talento_Tech_Backend.postman_collection.json`.
2. Click Import → File → select `Hackaton_Talento_Tech_Backend.postman_environment.json`.
3. In Postman, select the imported environment (Local - Hackaton Talento Tech Backend).
4. Edit environment `baseUrl` if your server runs in a different host or port (e.g., `http://localhost:8000`).
5. Set environment variable `file_path` to a valid local PDF path for the Upload PDF request (default `./pdfs/sample.pdf`).
6. Run requests manually or use Runner to run the whole collection.

Running the collection in the Runner:
1. Open Runner → choose collection `Hackaton Talento Tech - Backend` → choose environment `Local - Hackaton Talento Tech Backend`.
2. Click Start Run.

Notes:
- This backend does not expose `threads` endpoints or thread-based persistence by default — instead, use the `Chat (No thread)` or `Chat - Follow-up (no thread)` requests to send a standard chat request sequence.
- The tests in each request validate status codes and basic response shapes (answer exists) and set environment variables accordingly (where relevant).
- `Upload PDF` requests require a valid `file_path` variable in the environment for file uploads.

If you want to add more tests (e.g., content verification based on expected answers), you can edit the test scripts within Postman for each request.
