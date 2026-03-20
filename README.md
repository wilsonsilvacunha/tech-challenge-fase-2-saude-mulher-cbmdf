# tech-challenge-fase-2-saude-mulher-cbmdf

Sistema inteligente para apoio ao diagnóstico em saúde da mulher no contexto do CBMDF utilizando PLN e Machine Learning.

## Visão geral

Este repositório contém uma API em FastAPI para classificação de relatos curtos de sintomas em categorias diagnósticas simuladas. O objetivo atual é acadêmico: demonstrar um fluxo simples de PLN com `CountVectorizer` e `MultinomialNB`, exposto por HTTP.

## Estrutura principal

- `api/app.py`: camada HTTP com as rotas da API.
- `src/services/predictor.py`: serviço de inferência que carrega os artefatos treinados.
- `src/training/train_model.py`: script de treinamento e geração dos arquivos `.pkl`.
- `data/training_samples.csv`: dataset de exemplo usado para treinar o modelo atual.
- `models/modelo.pkl`: classificador serializado.
- `models/vectorizer.pkl`: vetorizador serializado.
- `tests/test_api.py`: testes automatizados do comportamento principal.

## Requisitos

- Python 3.13 ou compatível
- Ambiente virtual recomendado

Instalação das dependências:

```bash
pip install -r requirements.txt
```

## Como treinar novamente os artefatos

O modelo e o vetorizador podem ser recriados a partir de `data/training_samples.csv` com:

```bash
python -m src.training.train_model
```

Esse comando:

- lê o dataset CSV com colunas `texto` e `diagnostico`
- treina um `CountVectorizer`
- treina um `MultinomialNB`
- grava os artefatos em `models/modelo.pkl` e `models/vectorizer.pkl`

Formato esperado do dataset:

```csv
texto,diagnostico
"corrimento com odor forte","vaginose bacteriana"
"atraso menstrual com náusea","possível gravidez"
```

## Como executar a API

Na raiz do projeto:

```bash
uvicorn api.app:app --reload
```

Rotas disponíveis:

- `GET /`: verificação simples de funcionamento
- `POST /prever`: recebe um JSON com o campo `texto`

Exemplo de requisição:

```bash
curl -X POST http://127.0.0.1:8000/prever \
  -H "Content-Type: application/json" \
  -d '{"texto":"corrimento com odor forte"}'
```

Exemplo de resposta:

```json
{
  "diagnostico": "vaginose bacteriana"
}
```

## Testes

Para executar a suíte atual:

```bash
pytest tests/test_api.py -q
```

## Limitações

- O dataset atual é pequeno e simulado.
- O vocabulário do modelo é restrito e sensível à redação usada na entrada.
- O sistema não substitui avaliação clínica.
- O projeto ainda é um protótipo acadêmico, não uma ferramenta de produção.
