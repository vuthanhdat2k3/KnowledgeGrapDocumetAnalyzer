FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including pandoc, libreoffice, imagemagick (with EMF/WMF support)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pandoc \
    poppler-utils \
    libreoffice \
    imagemagick \
    libmagickwand-dev \
    ghostscript \
    libwmf-bin \
    libwmf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/sample_documents data/processed data/categories data/viewpoints data/markdown

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "src/ui/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
