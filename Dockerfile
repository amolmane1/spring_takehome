# FROM python:3.9-slim
# WORKDIR /app

# # 1) Install deps
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # 2) Copy pre-generated reference data
# COPY data/reference_data.json /app/data/reference_data.json

# # 3) Copy the rest of your code
# COPY . .

# # 4) Expose ports
# EXPOSE 80 8888

# # 5) Launch JupyterLab & FastAPI
# CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' & uvicorn app.main:app --host 0.0.0.0 --port 80"]
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
EXPOSE 80

# 5) Launch FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]