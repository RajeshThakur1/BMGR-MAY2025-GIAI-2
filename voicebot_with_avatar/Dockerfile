# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the specific livekit packages as requested
RUN pip install "livekit-agents[openai,cartesia,silero,turn-detector]~=1.0" "livekit-plugins-noise-cancellation~=0.2" "python-dotenv"

# Copy application files
COPY . .

# Create startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose ports
EXPOSE 5001 8080

# Use the startup script
CMD ["/start.sh"]
