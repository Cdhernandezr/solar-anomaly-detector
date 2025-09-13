FROM python:3.11-slim
WORKDIR /app
# Instala un conjunto común de dependencias para OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev
COPY requirements.txt .
# Actualiza pip y luego instala numpy primero
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.26.4

# Instala el resto de los requerimientos después, ignorando la versión de numpy del archivo
RUN pip install --no-cache-dir --no-deps -r requirements.txt
COPY ./api /app/api
COPY ./src /app/src
COPY ./models /app/models
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]