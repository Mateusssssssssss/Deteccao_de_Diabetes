from model_training import *

#Previsao
previsao = modelo.predict_proba(x_test)[:, 1]
print(previsao)


#  converte as probabilidades preditas em rótulos de classe (0 ou 1), com 0.5 como o limite para a classificação. 
# Se a probabilidade da classe 1 for maior que 0.5, a amostra será classificada como fraude (1), caso contrário, como não fraude (0).
prev_prob = (previsao > 0.7).astype(int)


