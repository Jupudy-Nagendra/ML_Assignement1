import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI  # Import FastAPI to create the application instance
from app.routes import router  # Import the router object containing API routes

# Create the FastAPI application instance
app = FastAPI()

# Include the router with all defined routes
app.include_router(router)

# Define a welcome route for the root endpoint
@app.get("/")
def welcome():
    """
    Welcome endpoint for the API.

    Returns:
        dict: A JSON message welcoming users to the API.
    """
    return {"message": "Welcome to the SVM Text Classification API"}