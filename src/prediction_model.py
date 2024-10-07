import joblib
from src.logger import get_logger

class PredictionModel:
    def __init__(self,todays_data):
        self.logger = get_logger()
        
        self.prediction = self.predict(todays_data)
        
    
    def predict(self,todays_data):
        loaded_model = joblib.load('models/best_model_logreg.pkl')
        self.logger.info("Model Loaded")


        prediction_tomorrow = loaded_model.predict(todays_data.fillna(value=0))
        pred = prediction_tomorrow[0]
        self.logger.info("Got the prediction")
