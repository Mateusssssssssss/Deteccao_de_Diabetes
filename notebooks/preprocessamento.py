from data.dados import *
from notebooks.eda import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


previsores = dados_codificados.iloc[:,0:8].values
classes = dados.iloc[:, 8].values

def pipeline_balanceamento(previsores, classes, undersample_ratio=0.1, oversample_ratio=0.5):
    """
    Aplica um pipeline de balanceamento com RandomUnderSampler e SMOTE.

    Parâmetros:
    - previsores: matriz com os dados de entrada (features)
    - classes: vetor com os rótulos (target)
    - undersample_ratio: proporção para subamostragem da classe majoritária
                         Ex: 0.1 = reduz a classe majoritária para 10% da minoria
    - oversample_ratio: proporção para superamostragem da classe minoritária
                        Ex: 0.7 = aumenta a minoria para 70% da nova maioria

    Retorna:
    - previsores_balanceados
    - classes_balanceadas
    """

    # Define os métodos de subamostragem e superamostragem
    majoritaria = RandomUnderSampler(sampling_strategy=undersample_ratio)
    minoritaria = SMOTE(sampling_strategy=oversample_ratio)

    # Cria o pipeline de balanceamento
    steps = [('maj', majoritaria), ('min', minoritaria)]
    pipeline = Pipeline(steps=steps)

    # Aplica o pipeline nos dados
    return pipeline.fit_resample(previsores, classes)


#Balanceamento dos dados
previsores, classes = pipeline_balanceamento(previsores, classes)

# Divisão dos dados em treino e teste
# 60% treino e 40% teste
x_train, x_test, y_train, y_test = train_test_split(previsores, classes, test_size=0.4, random_state=1)