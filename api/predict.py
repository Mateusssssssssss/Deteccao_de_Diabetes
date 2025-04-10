import joblib
from api.preprocess import preprocess_input

# Carrega modelo
pipeline = joblib.load("models/pipeline_diabetes.pkl")

def prediction(data):
    input_dict = data.model_dump()  # Converte o objeto Pydantic (InputData) em um dicionário.
    entrada = preprocess_input(input_dict)  # importação do preprocess_input para transformar os dados de entrada em um formato adequado para o modelo
    resultado = pipeline.predict_proba(entrada)  # Gera as probabilidades de cada classe com o modelo treinado
    prob = resultado[0][1]  # Captura a probabilidade da classe "1" (diabético)
    pred = int(prob > 0.7)  # Define a previsão: 1 se a probabilidade for maior que 0.7, senão 0

    return {
        "probabilidade": round(float(prob), 3),# 3 = 3 casas decimais
        "Resultado": 'Diabetico' if pred == 1 else 'Não Diabetico',# se pred == 1, retorna 'Diabetico', senão 'Não Diabetico'
    }