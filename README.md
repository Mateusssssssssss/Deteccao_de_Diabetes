# Detecção de Diabetes com XGBoost

Projeto de classificação para previsão de casos de diabetes com uso de Machine Learning. Utilizamos o modelo **XGBoostClassifier**, com técnicas de balanceamento de classes e análise exploratória de dados (EDA).

---

## Objetivo
Construir um modelo preditivo robusto capaz de detectar casos de diabetes a partir de um conjunto de dados clínicos, com máxima precisão e recall, especialmente para a classe minoritária (pacientes com diabetes).

---

## Estrutura do Projeto
```
project/
│
├── api/
│   ├── main.py               # Inicializa o FastAPI e importa o router
│   ├── models.py             # Define os dados de entrada com Pydantic
│   ├── predict.py            # Realiza a previsão com o modelo
│   ├── preprocess.py         # Função de pré-processamento dos dados
│   └── router.py             # Define a rota /predict da API
│
├── data/
│   ├── diabetes_prediction_dataset.csv  # Dataset original
│   ├── dados.py               # Carregamento e manipulação dos dados
│   ├── limpo.py               # Versão limpa ou tratada dos dados
│
├── notebooks/
│   ├── eda.py                 # Análise exploratória de dados
│   └── preprocessamento.py    # Balanceamento com SMOTE + undersampling, split treino/teste
│
├── src/
│   ├── model_training.py      # Treinamento com XGBoost
│   ├── model_evaluation.py    # Avaliação do modelo (classification_report, matriz de confusão)
│   └── predict.py             # Predição e conversão de probabilidades em classes
│
├── models/
│   ├── pipeline_diabetes.pkl  # Modelo final serializado
│   └── best_model.py          # Lógica para escolha e exportação do melhor modelo
│
└── README.md                  # Documentação do projeto
```

---

## Dataset
- Variáveis clínicas como idade, gênero, nível de glicose, histórico de tabagismo, etc.
- Sem valores nulos ou inconsistências encontradas após EDA.

---

## Modelagem
- **Modelo usado**: `XGBClassifier`
- **Técnicas aplicadas**:
  - Label Encoding (gênero, histórico de tabagismo)
  - SMOTE para oversampling da classe minoritária
  - RandomUnderSampler para undersampling da classe majoritária
  - Train/test split (60% treino, 40% teste)

---

## Parâmetros do Modelo
```python
XGBClassifier(
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
    scale_pos_weight=15,
)
```

---

## Métricas de Avaliação
```
Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.98      0.95      0.97     34135
           1       0.96      0.97      0.96     23665
```

- **Excelente performance em ambas as classes**
- **Recall para classe 1 (diabetes)**: 97%
- **F1-Score para classe 1**: 96%

---

## Tecnologias Utilizadas
- Python 3.10+
- XGBoost
- Pandas / NumPy
- Scikit-learn
- imbalanced-learn (SMOTE / Pipeline)
- Seaborn / Matplotlib

---

# API de Previsão de Diabetes com FastAPI

Esta API realiza previsões de diabetes com base em dados clínicos de pacientes, utilizando um modelo de machine learning treinado e integrado por meio da biblioteca FastAPI.

## Funcionalidades

- Previsão se um paciente tem diabetes ou não
- Retorna a probabilidade da previsão
- Utiliza modelo de machine learning serializado com `joblib`
- Pré-processamento de dados categóricos e numéricos
- Endpoint `POST` disponível para consumo por qualquer sistema

## Exemplo de Entrada

```json
{
  "gender": "Male",
  "age": 45,
  "hypertension": 1,
  "heart_disease": 0,
  "smoking_history": "former",
  "bmi": 28.7,
  "HbA1c_level": 6.5,
  "blood_glucose_level": 140
}
```

---

## Endpoint

### `POST /predict`

**Descrição:** Recebe os dados do paciente e retorna a probabilidade e o diagnóstico.

### Saída

```json
{
  "probabilidade": 0.823,
  "diabetico": "Diabetico"
}
```
### Erro

```json
{
  "detail": "Erro interno na previsão"
}
```

## Inicie o servidor

```bash
uvicorn main:app --reload
```

## Futuras Melhorias
- Teste com outras arquiteturas (Random Forest, Redes Neurais)
- API para servir o modelo via FastAPI ou Flask.
- Alguns ajustes no modelo



# Docker para Projetos Python

Este README fornece os comandos e passos essenciais para rodar e construir um projeto Python utilizando Docker.

## Requisitos

- Docker instalado na sua máquina.
- Um projeto Python com todos os arquivos necessários, como `Dockerfile`, `requirements.txt`, etc.

## Passos para usar o Docker

### 1. Criar o arquivo `Dockerfile`

Crie um arquivo chamado `Dockerfile` na raiz do seu projeto. Este arquivo contém as instruções para o Docker construir a imagem do seu projeto. Aqui está um exemplo básico:

```Dockerfile
# Use uma imagem base do Python
FROM python:3.12-slim

# Atualiza os pacotes do sistema e instala dependências do sistema
RUN apt-get update && apt-get install -y \ 
    gcc \ 
    libffi-dev \ 
    musl-dev \ 
    build-essential \ 
    && rm -rf /var/lib/apt/lists/*

# Instala o setuptools e wheel
RUN pip install --no-cache-dir setuptools wheel

# Copia os arquivos do projeto para o container
COPY . /app

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta que a aplicação usará
EXPOSE 8000

# Comando para rodar a API ou aplicação
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Criar o arquivo `requirements.txt`

Crie um arquivo `requirements.txt` com todas as dependências do seu projeto, por exemplo:

```
fastapi
uvicorn
numpy
pandas
scikit-learn
```

### 3. Construir a Imagem Docker

No terminal, vá até a pasta onde o `Dockerfile` está localizado e execute o seguinte comando para construir a imagem Docker:

```bash
docker build -t nome-da-imagem .
```

### 4. Rodar o Container

Depois de construir a imagem, você pode rodar um container com o comando:

```bash
docker run -p 8000:8000 nome-da-imagem
```

Isso vai rodar o seu container e expor a aplicação na porta 8000. Você pode acessar a API ou aplicação pelo navegador em `http://localhost:8000`.

### 5. Parar o Container

Para parar o container, use o seguinte comando:

```bash
docker stop nome-do-container
```

Onde `nome-do-container` é o nome ou ID do seu container (você pode pegar o ID com `docker ps`).

### 6. Remover a Imagem

Se você quiser remover a imagem após o uso, execute:

```bash
docker rmi nome-da-imagem
```

---

Esse é um exemplo básico de como configurar e rodar seu projeto Python no Docker. Você pode personalizar os comandos conforme a necessidade do seu projeto!




# Projeto com Docker Compose para API e MLflow

Este projeto utiliza o **Docker Compose** para rodar uma **API** de previsão de diabetes e o **MLflow** para o gerenciamento de experimentos e métricas do modelo de Machine Learning.

## Estrutura do Projeto

- **api**: Contém a API criada com **FastAPI** para previsão de diabetes.
- **mlflow**: Executa o servidor **MLflow** para rastrear e visualizar métricas de treinamento de modelos.

## Docker Compose

O **Docker Compose** é utilizado para orquestrar os containers da **API** e do **MLflow**, permitindo que ambos sejam executados em conjunto em um ambiente controlado.

## Como Rodar o Projeto

### Requisitos

- **Docker** instalado
- **Docker Compose** instalado

### Passos para Rodar

1. Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

2. Certifique-se de que você tem o arquivo **Dockerfile** e o **docker-compose.yml** na raiz do projeto.

3. Para rodar os containers com o Docker Compose, use o seguinte comando:

```bash
docker-compose up --build
```

- O comando `--build` garante que os containers serão construídos antes de serem executados.
- O **FastAPI** será acessível na porta `8000`.
- O **MLflow** será acessível na porta `5000`.

### O que Acontece Durante o Processo

- O **Docker Compose** irá criar e iniciar dois serviços:
  - **api**: A API FastAPI que faz previsões de diabetes. Ela depende do serviço **mlflow**.
  - **mlflow**: O servidor MLflow que irá rastrear e exibir as métricas dos experimentos.

### Acessos

- **API**: A API FastAPI estará disponível em `http://localhost:8000`.
- **MLflow**: O servidor MLflow estará disponível em `http://localhost:5000`.

### Parar os Containers

Para parar e remover os containers em execução, use o comando:

```bash
docker-compose down
```

Isso irá parar os serviços e limpar os containers, redes e volumes associados.


### Para rodar os containers novamente, basta usar o comando:

```bash
docker-compose up
```




# Publicando a Imagem Docker no Docker Hub

Este passo a passo mostra como você pode **autenticar, taguear e enviar sua imagem Docker para o Docker Hub**, deixando ela pronta para ser usada em ambientes como o Azure App Service.

---

## Verificando suas imagens locais

Antes de taguear ou enviar qualquer imagem, veja quais imagens já estão disponíveis localmente:

```bash
docker image 
```

📌 Esse comando mostra uma lista com o **REPOSITORY** (nome), **TAG** e **IMAGE ID** das imagens que você criou ou puxou.

---

## 1. Autenticação no Docker Hub

Antes de publicar a imagem, você precisa estar autenticado no Docker Hub.

```bash
docker login
```

🔸 Esse comando solicitará seu **nome de usuário** e **senha do Docker Hub**.  
🔸 Após o login bem-sucedido, você poderá interagir com seu repositório remoto (push/pull).

---

## 2. Taguear sua imagem local

Vamos dar um "rótulo" (tag) à imagem local para que ela aponte para o seu repositório no Docker Hub. O formato é:

```bash
docker tag <nome_local> <usuario_dockerhub>/<nome_imagem>:<tag>
```

Exemplo:

```bash
docker tag diabetes-api username/diabetes-api:latest
```

Aqui, `diabetes-api` é o nome da sua imagem local,  
`username` é seu usuário no Docker Hub,  
`latest` é a tag (versão da imagem — pode ser `v1`, `prod`, etc).

---

## 3. Enviando (push) a imagem para o Docker Hub

Com a imagem corretamente tagueada, execute:

```bash
docker push mateusgab/diabetes-api:latest
```

Isso enviará sua imagem para o repositório `username/diabetes-api` no Docker Hub.

Após o push ser concluído, sua imagem estará disponível publicamente (ou privada, se configurado assim) no seu repositório.