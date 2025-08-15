# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
# Assuming these are the dependencies based on main.py and settings.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8765 available to the world outside this container
EXPOSE 8765

# Run the application with full logging
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765", "--log-level", "debug", "--access-log"]
