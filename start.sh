#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Run FastAPI server with Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
