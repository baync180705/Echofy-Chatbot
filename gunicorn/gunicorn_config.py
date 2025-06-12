"""
Gunicorn configuration file for running the FastAPI application.

This file configures the number of workers, the worker class, and the binding
address for the Gunicorn server.
"""

import multiprocessing
from config import DEBUG

if(DEBUG):
    workers = 2 # Only 2 workers for Debug 
else:
    workers = multiprocessing.cpu_count() + 2 # Calculate number of workers as $num_cores + 2
worker_class = "uvicorn.workers.UvicornWorker" # Use Uvicorn worker class to run FastAPI
bind = "0.0.0.0:8000" # Bind to all interfaces on port 8000
