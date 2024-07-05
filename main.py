# Importing the predict_species function from the model module
from helpers.predict import predict_species 

from joblib import load
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

model = load('knn_iris_model.joblib')
labelEncoder = ('label_encoder.joblib')

# Defining a Pydantic model for the request payload
class inputData(BaseModel):
    #sample_input:list
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

@app.get("/")
async def root():
    return {"status":"Working!"}
    
@app.post("/predict")
async def predict(payload: inputData):
    print(payload.sepal_length)
    # Define new data for prediction
    new_data = np.array([
        [payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]
    ])

    encoded_predictions = predict_species(model, new_data)

    # Define a mapping from encoded values to species names
    species_mapping = {
            0: "setosa",
            1: "virginica",
            2: "versicolor" }

    # Display the predictions
    for i, species in enumerate(encoded_predictions):
        species_name = species_mapping.get(species, "Unknown species")
        print(f"Sample {i+1}: Predicted species is {species_name}")
    
    #return {"iris_species": str(encoded_predictions[0])}
    return {"iris_species": species_mapping.get(encoded_predictions[0], "Unknown species")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)