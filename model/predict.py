import joblib  # For loading the saved machine learning model

class ModelPredictor:
    """
    A class to handle model predictions for text input.

    Attributes:
        model (Pipeline): The loaded machine learning model.
    """

    def __init__(self, model_path: str):
        """
        Initialize the ModelPredictor by loading a trained model from the specified path.

        Args:
            model_path (str): Path to the saved model file.
        """
        self.model = joblib.load(model_path)  # Load the trained model using joblib

    def predict(self, text: str):
        """
        Make a prediction for a given text input using the loaded model.

        Args:
            text (str): The input text to classify.

        Returns:
            dict: A dictionary containing the prediction label and the associated probabilities.
                  - "prediction": Predicted label
                  - "probability": List of probabilities for each class
        """
        # Predict the label for the given text input
        prediction = self.model.predict([text])[0]
        
        # Predict the probabilities for each class
        probability = self.model.predict_proba([text])[0]
        
        # Return the prediction and probabilities as a dictionary
        return {
            "prediction": prediction, 
            "probability": probability.tolist()  # Convert to list for better usability
        }