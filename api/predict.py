import joblib
from api.preprocess import preprocess_input

# Carrega modelo
pipeline = joblib.load("models/pipeline_diabetes.pkl")

def prediction(data):
    input_dict = data.model_dump()
    entrada = preprocess_input(input_dict)
    resultado = pipeline.predict_proba(entrada)
    prob = resultado[0][1]
    pred = int(prob > 0.7)
    return {
        "probabilidade": round(float(prob), 3),
        "diabetico": 'Diabetico' if pred == 1 else 'Não Diabetico',
    }