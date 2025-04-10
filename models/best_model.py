import joblib
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score

# Importa dados e EDA
from notebooks.eda import *

# Define previsores e classes
previsores = dados_codificados.iloc[:, 0:8].values
classes = dados.iloc[:, 8].values

# Etapas do pipeline de balanceamento + modelo
balanceamento_modelo_pipeline = Pipeline(steps=[
    ("undersample", RandomUnderSampler(sampling_strategy=0.1)),
    ("oversample", SMOTE(sampling_strategy=0.5)),
    ("modelo", XGBClassifier(
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
        scale_pos_weight=15
    ))
])

# Divide os dados
x_train, x_test, y_train, y_test = train_test_split(previsores, classes, test_size=0.4, random_state=1)

# Validação cruzada
results = cross_val_score(balanceamento_modelo_pipeline, x_train, y_train, cv=5)
print(f'Cross Validation: {results}')

# Treinamento final
balanceamento_modelo_pipeline.fit(x_train, y_train)

# Salvar o pipeline completo
joblib.dump(balanceamento_modelo_pipeline, 'pipeline_diabetes.pkl')

