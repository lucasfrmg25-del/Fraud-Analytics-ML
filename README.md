# Fraud Analytics ML

> **Quando 99,59% de acurácia não significa um bom modelo.**

Projeto de Fraud Analytics desenvolvido para demonstrar, de forma prática, como **estatística, desbalanceamento de classes e escolha de métricas** podem alterar a interpretação de um modelo de Machine Learning e, consequentemente, a tomada de decisão em prevenção a fraudes.

O objetivo deste estudo não é construir o detector de fraude mais complexo possível. A proposta é isolar uma decisão estatística simples e observar seu impacto: comparar duas Regressões Logísticas treinadas sobre a mesma feature, a mesma amostra e a mesma divisão de treino e teste, alterando apenas o tratamento dado ao desbalanceamento das classes.

## Objetivo

Em bases de fraude, eventos fraudulentos normalmente representam uma parcela muito pequena do total de transações. Nesse cenário, avaliar um modelo apenas pela acurácia pode levar a conclusões equivocadas.

Este projeto busca responder:

**Como o balanceamento das classes pode alterar a capacidade de detectar fraudes e a decisão operacional, mesmo sem adicionar novas variáveis ao modelo?**

## Dataset

O estudo utiliza o arquivo `fraudTest.csv`, com **555.719 transações** e **23 colunas**.

A variável alvo é:

- `is_fraud = 0`: transação legítima
- `is_fraud = 1`: fraude

Para manter o experimento simples e facilitar a interpretação do efeito estatístico, os dois modelos utilizam somente a feature `amt` — valor da transação.

## Análise exploratória

A análise inicial encontrou:

| Indicador | Resultado |
|---|---:|
| Total de transações | 555.719 |
| Fraudes | 2.145 |
| Taxa de fraude | **0,39%** |
| Volume financeiro total | R$ 38.562.903,11 |
| Volume financeiro fraudado | R$ 1.133.324,68 |
| Ticket médio geral | R$ 69,39 |
| Ticket médio das fraudes | **R$ 528,36** |
| Participação da fraude no volume financeiro | **2,94%** |

Embora apenas **0,39% das transações** sejam fraudulentas, elas representam **2,94% do volume financeiro** analisado.

Além disso, o ticket médio das fraudes é aproximadamente **7,6 vezes** o ticket médio geral.

Esses números já evidenciam duas características relevantes: forte desbalanceamento entre as classes e impacto financeiro desproporcional dos eventos fraudulentos.

## Preparação dos dados

A feature e o target foram definidos como:

```python
X = df[["amt"]]
y = df["is_fraud"]
```

A base foi dividida em **80% para treinamento e 20% para teste**, preservando a proporção das classes com `stratify=y`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

O uso da mesma divisão nos dois modelos permite comparar os resultados sob as mesmas condições.

## Modelo 1 — Regressão Logística sem balanceamento

O primeiro modelo foi utilizado como **baseline**:

```python
Lr = LogisticRegression()
Lr.fit(X_train, y_train)

y_pred = Lr.predict(X_test)
matriz = confusion_matrix(y_test, y_pred)
```

### Matriz de confusão

```text
[[110688     27]
 [   429      0]]
```

Interpretação:

| Resultado | Quantidade |
|---|---:|
| Legítimas classificadas corretamente | 110.688 |
| Legítimas classificadas como fraude | 27 |
| Fraudes não detectadas | **429** |
| Fraudes detectadas | **0** |

A acurácia fica em aproximadamente **99,59%**. Entretanto, o modelo não identificou nenhuma das 429 fraudes existentes no conjunto de teste.

**Recall da classe fraude: 0%.**

Esse resultado demonstra por que uma métrica isolada pode ser enganosa: um modelo pode apresentar acurácia extremamente alta e, ainda assim, falhar completamente no objetivo de negócio.

## Modelo 2 — Regressão Logística com classes balanceadas

No segundo experimento, mantivemos:

- o mesmo dataset;
- a mesma feature `amt`;
- a mesma divisão treino/teste;
- o mesmo algoritmo.

A mudança foi apenas a inclusão de:

```python
class_weight="balanced"
```

Modelo:

```python
Lr_balanced = LogisticRegression(class_weight="balanced")
Lr_balanced.fit(X_train, y_train)

y_pred_balanced = Lr_balanced.predict(X_test)
matriz_balanced = confusion_matrix(y_test, y_pred_balanced)
```

O parâmetro `class_weight="balanced"` ajusta automaticamente os pesos das classes de forma inversamente proporcional às suas frequências, aumentando a importância dos exemplos pertencentes à classe minoritária durante o treinamento.

### Matriz de confusão

```text
[[105474   5241]
 [   111    318]]
```

Interpretação:

| Resultado | Quantidade |
|---|---:|
| Legítimas classificadas corretamente | 105.474 |
| Legítimas classificadas como fraude | 5.241 |
| Fraudes não detectadas | 111 |
| Fraudes detectadas | **318** |

O modelo passou a detectar **318 das 429 fraudes** presentes no teste.

**Recall da classe fraude: aproximadamente 74,13%.**

A acurácia caiu para aproximadamente **95,19%**, mas a capacidade de identificar fraudes aumentou de **0% para 74,13%**.

## Comparação dos experimentos

| Indicador | Modelo 1 | Modelo 2 |
|---|---:|---:|
| Balanceamento | Não | Sim |
| Feature utilizada | `amt` | `amt` |
| Fraudes reais no teste | 429 | 429 |
| Fraudes detectadas | **0** | **318** |
| Fraudes não detectadas | 429 | 111 |
| Falsos positivos | 27 | 5.241 |
| Recall de fraude | **0%** | **74,13%** |
| Acurácia aproximada | **99,59%** | **95,19%** |

## O ponto central: estatística e decisão

A principal conclusão não é que o Modelo 2 seja perfeito.

O experimento mostra que uma alteração estatística simples foi suficiente para modificar radicalmente o comportamento do modelo.

No primeiro cenário, a alta acurácia escondia uma incapacidade completa de detectar a classe de maior interesse. No segundo, o modelo passou a identificar aproximadamente três quartos das fraudes, porém ao custo de um aumento relevante nos falsos positivos.

Em uma operação real de prevenção a fraudes, isso representa um **trade-off de negócio**:

- maior sensibilidade pode reduzir perdas financeiras;
- mais falsos positivos podem aumentar filas de análise e atrito com clientes legítimos;
- a escolha do modelo depende do custo e do risco associados a cada tipo de erro.

Portanto, a pergunta não deve ser apenas **"qual modelo possui maior acurácia?"**, mas:

**Qual comportamento estatístico atende melhor à decisão que o negócio precisa tomar?**

## Pipeline do projeto

```text
Dados
  ↓
Análise exploratória
  ↓
KPIs de fraude
  ↓
Definição de X e y
  ↓
Train / Test Split
  ↓
Modelo 1 — Logistic Regression
  ↓
Diagnóstico do desbalanceamento
  ↓
Modelo 2 — Logistic Regression + class_weight="balanced"
  ↓
Matriz de confusão
  ↓
Comparação estatística
  ↓
Interpretação de negócio
```

## Tecnologias

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## Principais aprendizados

Este estudo reforça que, em Fraud Analytics, **modelar não é apenas buscar o maior percentual de acerto**. É necessário compreender a distribuição dos dados, selecionar métricas coerentes com o problema e traduzir o comportamento estatístico do modelo em impacto operacional.

A evolução de **0% para 74,13% de Recall** sem alterar o algoritmo ou adicionar novas features demonstra de forma prática como o tratamento estatístico das classes pode influenciar a tomada de decisão.

---

### Lucas Freire Martins | Fraud Analytics

**Dados → Estatística → Modelo → Decisão → Impacto**
