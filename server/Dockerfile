FROM python:3.11-slim

# Required env vars
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install server dependencies first (Docker cache optimization)
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy the full project
COPY . /app

# Install the permanence package itself
RUN pip install --no-cache-dir -e /app

# Expose HuggingFace Spaces standard port
EXPOSE 7860

# Health check — HuggingFace pings this during Space validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()" || exit 1

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
