from joblib import load

def load_model(model_path):
    model = load(model_path)
    return model

def load_labelEncoder(encoder_path):
    label_encoder = load(encoder_path)
    return label_encoder