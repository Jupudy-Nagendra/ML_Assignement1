import time
from functools import wraps

def monitor_prediction_time(func):
    """
    Decorator to monitor the execution time of the prediction function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        
        execution_time = end_time - start_time  # Calculate the time difference
        print(f"Prediction time: {execution_time:.4f} seconds")  # Print the time taken
        
        return result  # Return the result of the original function
    
    return wrapper