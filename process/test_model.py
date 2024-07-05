from joblib import load

# Load the saved KNN model
filename = "knn_iris_model.joblib"
knn_loaded = load(filename)
label_encoder_path = 'label_encoder.joblib'
label_encoder = load(label_encoder_path)

# Assuming the model is already loaded from the saved file 'knn_iris_model_with_label_encoder.joblib'
# Assuming the model is already loaded from the saved file 'knn_iris_model_with_label_encoder.joblib'

# Predict new data samples
new_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]
encoded_predictions = knn_loaded.predict(new_data)

# Decode the encoded predictions back to original class labels
decoded_predictions = label_encoder.classes_[encoded_predictions]

print("Predictions for new data samples:", decoded_predictions)
