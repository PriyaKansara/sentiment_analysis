# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port FastAPI runs on
EXPOSE 8000

# Command to run FastAPI using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
