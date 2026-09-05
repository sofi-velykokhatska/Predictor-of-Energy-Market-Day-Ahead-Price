# Use slim Python image to keep container small
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY models/ models/

# Copy API code
COPY energy_api.py .

# Expose port 8080
EXPOSE 8080

# Health check - Docker will monitor if container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API
CMD ["python", "energy_api.py"]
