FROM python:3.9-slim
WORKDIR /app

# 1) Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy pre-generated reference data
COPY data/reference_data.json /app/data/reference_data.json

# 3) Copy the rest of your code
COPY . .

# 4) Expose FastAPI port
# EXPOSE 8080
EXPOSE 10000

# 5) Launch FastAPI
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]