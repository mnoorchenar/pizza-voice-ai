FROM python:3.10-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 7860
CMD ["gunicorn","--bind","0.0.0.0:7860","--workers","2","--timeout","120","app:app"]