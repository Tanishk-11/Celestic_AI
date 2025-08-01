# Dockerfile

# Start with a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container first
# This helps with Docker's layer caching, speeding up future builds
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
# This includes app.py, your .keras model, and the static/ and templates/ folders
COPY . .

# Expose the port that the application will run on
EXPOSE 5000

# Define the command to run your application using Gunicorn
# Gunicorn is a production-ready web server for Python applications
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]
