from pydantic import BaseModel

# Classe que define o formato dos dados de entrada da API.
# Usada para validação automática com Pydantic e documentação no Swagger.
# Garante que o modelo receba os dados no tipo e formato corretos:
class InputData(BaseModel): 
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
