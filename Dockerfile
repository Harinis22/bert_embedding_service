FROM python:3.11-slim  # Use Python 3.11 for compatibility; update when 3.13 is officially supported

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Streamlit's default port
EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
