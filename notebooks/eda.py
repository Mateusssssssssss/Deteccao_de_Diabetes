import sys
import os

# Adiciona o diretório raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import seaborn as sb
import matplotlib.pyplot as plt
from data.dados import load_data 
from sklearn.preprocessing import LabelEncoder

dados = load_data()
print(dados.describe())
print(dados.head())
print(dados.shape)

# Verifica valores nulos
null = dados.isnull().sum()
print(f'Valores Nulos: {null}')

# Verifica duplicatas
duplicados = dados.duplicated().sum()
print(f'Duplicados: {duplicados}')

# Boxplot para análise de outliers
sb.boxplot(dados['blood_glucose_level'])
plt.show()

#Quantidade de fraudes
diabetico = (dados['diabetes'] == 1).sum()
print(f'Quantidade de Diabeticos: {diabetico}')

#Quantidade de não fraudes
nao_diabetico = (dados['diabetes'] == 0).sum()
print(f'Quantidade de Não Diabeticos: {nao_diabetico}')
# Cria uma cópia para não alterar o original (opcional)
dados_codificados = dados.copy()

# Codifica as colunas categóricas
labelencoder = LabelEncoder()
dados_codificados['gender'] = labelencoder.fit_transform(dados_codificados['gender'])
dados_codificados['smoking_history'] = labelencoder.fit_transform(dados_codificados['smoking_history'])


# correlação numérica
correlacao = dados_codificados.corr()['diabetes']
print(correlacao)
# Calcula a correlação com a variável alvo
correlacoes = dados_codificados.corr()['diabetes'].drop('diabetes')

# Pega as 5 variáveis com maior correlação
top_five = correlacoes.sort_values(ascending=False).head(5)
print(f'top 5: {top_five}')
top_five = correlacoes.sort_values(ascending=False).head(5).index.tolist()

#Verificar quais generos possuem na tabela gender.
outros = dados['gender'].unique()
print(f'Quais Generos possuem: {outros}')


