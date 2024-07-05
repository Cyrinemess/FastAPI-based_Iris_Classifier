
def predict_species(model, new_data):
    """
    Predict the species for new data using the loaded model and label encoder.
    
    Parameters:
    - model: The loaded classifier model.
    - labelEncoder: The loaded label encoder.
    - new_data (numpy.ndarray): A 2D array where each row represents a sample.
    
    Returns:
    - List of predicted species names.
    """
    predictions = model.predict(new_data)
    #predicted_species = labelEncoder.inverse_transform(predictions)
    return predictions
