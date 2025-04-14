from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from notebooks.eda import *
from notebooks.preprocess import *

modelo = XGBClassifier(objective='binary:logistic',  # Classificação binária
    eval_metric='logloss',            # Métrica de avaliação
    n_estimators=500,             # Número de árvores
    learning_rate=0.05,           # Taxa de aprendizado
    max_depth=20,                  # Profundidade das árvores
    subsample=0.5,                # Amostragem para evitar overfitting
    colsample_bytree=0.7,         # Porcentagem de colunas usadas
    gamma=1,                      # Evita overfitting
    reg_lambda=0,                 # Regularização L2
    reg_alpha=1,                   # Regularização L1
    scale_pos_weight=30,             # dá mais peso para a classe minoritária
)

#Validação cruzada
results = cross_val_score(modelo, x_train, y_train, cv=5) 
print(f'Cross Validation: {results}')

modelo.fit(x_train, y_train)



# # Salva o modelo treinado
# joblib.dump(modelo, "modelo_diabetes.pkl")

