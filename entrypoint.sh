#!/bin/bash

# Run Hugging Face login once
python -m startup.hf_login

# Populate the DB
python -m startup.db

# Now start the Gunicorn server
exec gunicorn main:app -c gunicorn_config.py
