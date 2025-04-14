import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from notebooks.preprocess import previsores, classes 
import matplotlib.pyplot as plt
import tempfile
import os

# Define o servidor de tracking do MLflow
# Define o endereço do servidor MLflow (vem das variáveis de ambiente do Docker)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("previsao_diabetes_xgboost")


# Separação dos dados
x_train, x_test, y_train, y_test = train_test_split(previsores, classes, test_size=0.4, random_state=1)

# Hiperparâmetros
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 20,
    "subsample": 0.5,
    "colsample_bytree": 0.7,
    "gamma": 1,
    "reg_lambda": 0,
    "reg_alpha": 1,
    "scale_pos_weight": 30
}

# Início do experimento MLflow
with mlflow.start_run():
    print("Iniciando o experimento...")
    modelo = XGBClassifier(**params)
    modelo.fit(x_train, y_train)
    predictions = modelo.predict_proba(x_test)[:, 1]
    prev_prob = (predictions > 0.7).astype(int)

    # Métricas
    acc = accuracy_score(y_test, prev_prob)
    prec = precision_score(y_test, prev_prob)
    rec = recall_score(y_test, prev_prob)
    f1 = f1_score(y_test, prev_prob)

    # Logando hiperparâmetros
    for key, value in params.items():
        mlflow.log_param(key, value)

    # Logando métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Matriz de confusão
    cm = confusion_matrix(y_test, prev_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
    disp.plot(cmap='Blues')

    # Salva a imagem temporariamente
    with tempfile.TemporaryDirectory() as temp_dir:
        fig_path = os.path.join(temp_dir, "confusion_matrix.png")
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path, "confusion_matrix")
        plt.close()

    # Salvando o modelo
    mlflow.sklearn.log_model(modelo, "modelo_xgb")

    print(f"Modelo logado com accuracy: {acc}")
