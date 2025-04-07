import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from data.bruto import dados
from notebooks.eda import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler



previsores = dados_codificados.iloc[:,0:8].values
classes = dados.iloc[:, 8].values



#RandomUnderSampler(sampling_strategy=0.1): Isso significa que a classe majoritária 
# será reduzida para 10% do tamanho original da classe minoritária
majoritaria = RandomUnderSampler(sampling_strategy = 0.1)

# SMOTE(sampling_strategy=0.5): aumentando a classe minoritária para 50% 
# do tamanho da classe majoritária, ou seja, a classe minoritária será aumentada para 
# ser 1,5 vezes maior que a classe majoritária após o balanceamento.
minioritaria = SMOTE(sampling_strategy=0.7)
steps = [('maj', majoritaria),('min', minioritaria)]
pipeline = Pipeline(steps=steps)
previsores, classes = pipeline.fit_resample(previsores, classes)

x_train, x_test, y_train, y_test = train_test_split(previsores, classes, test_size=0.4, random_state=1)

modelo = XGBClassifier(objective='binary:logistic',  # Classificação binária
    eval_metric='logloss',            # Métrica de avaliação
    n_estimators=400,             # Número de árvores
    learning_rate=0.04,           # Taxa de aprendizado
    max_depth=15,                  # Profundidade das árvores
    subsample=0.5,                # Amostragem para evitar overfitting
    colsample_bytree=0.6,         # Porcentagem de colunas usadas
    gamma=1,                      # Evita overfitting
    reg_lambda=1,                 # Regularização L2
    reg_alpha=1,                   # Regularização L1
    scale_pos_weight=7,             # dá mais peso para a classe minoritária
)


modelo.fit(x_train, y_train)
