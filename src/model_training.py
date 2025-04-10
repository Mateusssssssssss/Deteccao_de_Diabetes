import sys
import os

# Adiciona o diretório raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xgboost import XGBClassifier
from notebooks.eda import *
from notebooks.preprocessamento import *
from sklearn.model_selection import cross_val_score



modelo = XGBClassifier(objective='binary:logistic',  # Classificação binária
    eval_metric='logloss',            # Métrica de avaliação
    n_estimators=400,             # Número de árvores
    learning_rate=0.05,           # Taxa de aprendizado
    max_depth=15,                  # Profundidade das árvores
    subsample=0.5,                # Amostragem para evitar overfitting
    colsample_bytree=0.5,         # Porcentagem de colunas usadas
    gamma=1,                      # Evita overfitting
    reg_lambda=0,                 # Regularização L2
    reg_alpha=1,                   # Regularização L1
    scale_pos_weight=15,             # dá mais peso para a classe minoritária
)

#Validação cruzada
results = cross_val_score(modelo, x_train, y_train, cv=5) 
print(f'Cross Validation: {results}')

modelo.fit(x_train, y_train)



# # Salva o modelo treinado
# joblib.dump(modelo, "modelo_diabetes.pkl")

