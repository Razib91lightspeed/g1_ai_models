FROM g1-qwen-vlm:0.1

WORKDIR /srv/app

RUN python -m pip install --no-cache-dir \
    "fastapi>=0.115,<1" \
    "uvicorn>=0.30,<1"

COPY server.py .

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
