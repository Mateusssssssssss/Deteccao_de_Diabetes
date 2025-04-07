from model_training import * 
from sklearn.metrics import classification_report,confusion_matrix, average_precision_score

#Previsao
previsao = modelo.predict_proba(x_test)[:, 1]
print(previsao)
#  converte as probabilidades preditas em rótulos de classe (0 ou 1), com 0.5 como o limite para a classificação. 
# Se a probabilidade da classe 1 for maior que 0.5, a amostra será classificada como fraude (1), caso contrário, como não fraude (0).
prev_prob = (previsao > 0.5).astype(int)


# Calcular AUPRC
# AUPRC é útil para modelos em dados desbalanceados e mede a área sob a curva de 
# precisão vs. recall. Quanto maior a AUPRC, melhor o modelo!
auprc = average_precision_score(y_test, prev_prob)
print(f'Auprc: {auprc}')

#weighted avg: A média ponderada leva em consideração o número de amostras de cada classe, 
# ou seja, ela ajusta as métricas com base na proporção das classes no conjunto de dados
scores = classification_report(y_test, prev_prob)
print(f'Scores: {scores}')

#Dados FP FV VP VF
confusion = confusion_matrix(y_test, prev_prob)
print(f'Confusão: {confusion}')