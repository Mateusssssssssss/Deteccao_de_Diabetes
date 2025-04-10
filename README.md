# Detecção de Diabetes com XGBoost

Projeto de classificação para previsão de casos de diabetes com uso de Machine Learning. Utilizamos o modelo **XGBoostClassifier**, com técnicas de balanceamento de classes e análise exploratória de dados (EDA).

---

## Objetivo
Construir um modelo preditivo robusto capaz de detectar casos de diabetes a partir de um conjunto de dados clínicos, com máxima precisão e recall, especialmente para a classe minoritária (pacientes com diabetes).

---

## Estrutura do Projeto
```
project/
│
├── api/
│   ├── main.py               # Inicializa o FastAPI e importa o router
│   ├── models.py             # Define os dados de entrada com Pydantic
│   ├── predict.py            # Realiza a previsão com o modelo
│   ├── preprocess.py         # Função de pré-processamento dos dados
│   └── router.py             # Define a rota /predict da API
│
├── data/
│   ├── diabetes_prediction_dataset.csv  # Dataset original
│   ├── dados.py               # Carregamento e manipulação dos dados
│   ├── limpo.py               # Versão limpa ou tratada dos dados
│
├── notebooks/
│   ├── eda.py                 # Análise exploratória de dados
│   └── preprocessamento.py    # Balanceamento com SMOTE + undersampling, split treino/teste
│
├── src/
│   ├── model_training.py      # Treinamento com XGBoost
│   ├── model_evaluation.py    # Avaliação do modelo (classification_report, matriz de confusão)
│   └── predict.py             # Predição e conversão de probabilidades em classes
│
├── models/
│   ├── pipeline_diabetes.pkl  # Modelo final serializado
│   └── best_model.py          # Lógica para escolha e exportação do melhor modelo
│
└── README.md                  # Documentação do projeto
```

---

## Dataset
- Variáveis clínicas como idade, gênero, nível de glicose, histórico de tabagismo, etc.
- Sem valores nulos ou inconsistências encontradas após EDA.

---

## Modelagem
- **Modelo usado**: `XGBClassifier`
- **Técnicas aplicadas**:
  - Label Encoding (gênero, histórico de tabagismo)
  - SMOTE para oversampling da classe minoritária
  - RandomUnderSampler para undersampling da classe majoritária
  - Train/test split (60% treino, 40% teste)

---

## Parâmetros do Modelo
```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=400,
    learning_rate=0.05,
    max_depth=15,
    subsample=0.5,
    colsample_bytree=0.5,
    gamma=1,
    reg_lambda=0,
    reg_alpha=1,
    scale_pos_weight=15,
)
```

---

## Métricas de Avaliação
```
Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.98      0.95      0.97     34135
           1       0.96      0.97      0.96     23665
```

- **Excelente performance em ambas as classes**
- **Recall para classe 1 (diabetes)**: 97%
- **F1-Score para classe 1**: 96%

---

## Tecnologias Utilizadas
- Python 3.10+
- XGBoost
- Pandas / NumPy
- Scikit-learn
- imbalanced-learn (SMOTE / Pipeline)
- Seaborn / Matplotlib

---

# API de Previsão de Diabetes com FastAPI

Esta API realiza previsões de diabetes com base em dados clínicos de pacientes, utilizando um modelo de machine learning treinado e integrado por meio da biblioteca FastAPI.

## Funcionalidades

- Previsão se um paciente tem diabetes ou não
- Retorna a probabilidade da previsão
- Utiliza modelo de machine learning serializado com `joblib`
- Pré-processamento de dados categóricos e numéricos
- Endpoint `POST` disponível para consumo por qualquer sistema

## Exemplo de Entrada

```json
{
  "gender": "Male",
  "age": 45,
  "hypertension": 1,
  "heart_disease": 0,
  "smoking_history": "former",
  "bmi": 28.7,
  "HbA1c_level": 6.5,
  "blood_glucose_level": 140
}
```

---

## Endpoint

### `POST /predict`

**Descrição:** Recebe os dados do paciente e retorna a probabilidade e o diagnóstico.

### Saída

```json
{
  "probabilidade": 0.823,
  "diabetico": "Diabetico"
}
```
### Erro

```json
{
  "detail": "Erro interno na previsão"
}
```

## Inicie o servidor

```bash
uvicorn main:app --reload
```

## Futuras Melhorias
- Teste com outras arquiteturas (Random Forest, Redes Neurais)
- API para servir o modelo via FastAPI ou Flask.
- Alguns ajustes no modelo



