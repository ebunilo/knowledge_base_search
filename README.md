# Knowledge Base Search

Enterprise knowledge-base retrieval and grounded Q&A with a web UI.

## Run with Docker Compose

1. Copy environment template and fill secrets:

   ```bash
   cp .env.example .env
   ```

2. Start the stack:

   ```bash
   docker compose up -d --build --wait
   ```

3. Open the app:

   - `http://localhost:8765` (local machine)
   - `http://<server-ip>:8765` (remote host with firewall rules open)

## Stop

```bash
docker compose down
```
