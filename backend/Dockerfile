FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (including build tools for compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglx-mesa0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first
COPY requirements.txt .

# Upgrade pip and install dependencies using Tsinghua mirror
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --default-timeout=1000

# Copy application code (changes frequently, so last)
COPY . .

# Precompile Python files for faster startup
RUN python -m compileall .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
