import sys
import os

# Adiciona o diretório raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import classification_report,confusion_matrix, average_precision_score
from notebooks.preprocessamento import *
from predict import *

def metricas(y_true, y_pred):
    """
    Avalia o desempenho de um modelo de classificação com foco em dados desbalanceados.

    Parâmetros:
    - y_true: valores reais (verdadeiros) do conjunto de teste.
    - y_pred: previsões do modelo (pode ser probabilidade ou classe, depende do uso).

    Exibe:
    - AUPRC (Área sob a Curva de Precisão-Recall)
    - Classification Report (com médias ponderadas)
    - Matriz de Confusão
    """
    # AUPRC - ideal para dados desbalanceados
    auprc = average_precision_score(y_true, y_pred)
    print(f'AUPRC: {auprc}')

    # Classification report - inclui precisão, recall, f1-score.
    scores = classification_report(y_true, y_pred)
    print('Relatório de Classificação:')
    print(scores)

    # Matriz de confusão - mostra FP, FN, VP, VN
    confusion = confusion_matrix(y_true, y_pred)
    print('Matriz de Confusão:')
    print(confusion)



# metricas
metricas(y_test, prev_prob)