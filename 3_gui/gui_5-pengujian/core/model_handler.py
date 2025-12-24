import numpy as np
from tensorflow import keras

class ModelHandler:
    def __init__(self):
        self.model = None
        
    def load_model(self, model_path):
        try:
            self.model = keras.models.load_model(model_path)
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    def predict(self, data):
        if self.model is None:
            raise Exception("Model not loaded")
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1, 1)
        elif len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        
        prediction = self.model.predict(data, verbose=0)
        binary_pred = (prediction > 0.5).astype(int).flatten()
        
        return binary_pred