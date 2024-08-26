# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt y lo instala
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de tu proyecto al contenedor
COPY . .

# Comando para ejecutar el proceso batch
CMD ["python", "src/evaluation.py"]
