FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# We map port 8000 for the FastAPI server
EXPOSE 8000

# Start Uvicorn, listening on all interfaces (0.0.0.0) so Docker can map the port
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
