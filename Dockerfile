# Dockerfile

# Use an official Python 3.11 runtime as a parent image, as specified in runtime.txt
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code and model files into the container
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the FastAPI application with Uvicorn when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
