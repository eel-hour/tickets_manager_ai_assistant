# Dockerfile (at project root)

FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data folder
COPY streamlit_app.py .
COPY data ./data

# Expose Streamlit port
EXPOSE 8501

# Launch the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

