# Use the official Python image as the base image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code to the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Make the Entrypoint executable
RUN chmod +x entrypoint.sh

# Execute the Entrypoint script
ENTRYPOINT ["./entrypoint.sh"]