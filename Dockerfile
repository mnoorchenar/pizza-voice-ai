FROM python:3.11-slim

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Upgrade pip once, then install deps — separate layer so it's cached on rebuilds
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files and hand ownership to non-root user
COPY --chown=user:user . .

# Switch to non-root user AFTER install
USER user

EXPOSE 7860

CMD ["python", "app.py"]