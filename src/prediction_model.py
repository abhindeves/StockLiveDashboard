import joblib
from src.logger import get_logger

class PredictionModel:
    def __init__(self, todays_data):
        self.logger = get_logger()
        self.logger.info("Initializing PredictionModel with provided data.")
        self.prediction = self.predict(todays_data)

    def predict(self, todays_data):
        self.logger.info("Loading the model from 'models/best_model_logreg.pkl'.")
        try:
            loaded_model = joblib.load('models/best_model_logreg.pkl')
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

        self.logger.info("Making predictions on the provided data.")
        prediction_tomorrow = loaded_model.predict(todays_data.fillna(value=0))
        pred = prediction_tomorrow[0]
        self.logger.info("Prediction made successfully.")
        return pred