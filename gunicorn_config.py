import multiprocessing

# Calculate number of workers as $num_cores + 2
workers = multiprocessing.cpu_count() + 2

# Use Uvicorn worker class to run FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Bind to all interfaces on port 8000
bind = "0.0.0.0:8000"

