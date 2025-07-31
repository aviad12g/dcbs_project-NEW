# Use Python 3.11 slim image for efficiency
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p results temp_cache checkpoints mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/temp_cache
ENV HF_HOME=/app/temp_cache

# Expose port for Streamlit (optional)
EXPOSE 8501

# Default command (can be overridden)
CMD ["python", "compare_methods.py", "--limit", "5", "--samplers", "greedy", "dcbs"]