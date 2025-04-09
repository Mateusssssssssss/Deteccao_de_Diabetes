import pandas as pd

def load_data():
    """
    Função para carregar o dataset de diabetes.
    """
    # Carregar o dataset
    dados = pd.read_csv('data/diabetes_prediction_dataset.csv')
    
    # Retornar o dataframe
    return dados

if __name__ == '__main__':
    dados = pd.read_csv('data/diabetes_prediction_dataset.csv')
    print(dados.head())

