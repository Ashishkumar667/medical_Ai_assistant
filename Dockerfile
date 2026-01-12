# ----------------------------
# Base image (small & stable)
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# Environment settings
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Set work directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Install Python dependencies
# ----------------------------
COPY requirements.txt .

# Force CPU-only PyTorch (CRITICAL)
RUN pip install --upgrade pip && \
    pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# ----------------------------
# Copy application code
# ----------------------------
COPY . .

# ----------------------------
# Expose port (Render requirement)
# ----------------------------
EXPOSE 10000

# ----------------------------
# Start FastAPI app
# ----------------------------
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
