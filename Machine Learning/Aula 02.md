# Aula 02 – NumPy e Pandas para Machine Learning

## Preparando os Dados para a Inteligência Artificial

**Carga Horária:** 4 horas

---

# Objetivos da Aula

Ao final desta aula você será capaz de:

✅ Entender o que é NumPy.

✅ Criar vetores e matrizes.

✅ Manipular dados utilizando Pandas.

✅ Ler arquivos CSV.

✅ Filtrar informações.

✅ Tratar dados ausentes.

✅ Preparar dados para Machine Learning.

---

# Relembrando a Aula Anterior

Na aula passada aprendemos:

```text
Dados → Algoritmo → Modelo → Previsão
```

Mas existe um problema...

Na vida real os dados nunca vêm organizados.

Imagine uma planilha assim:

| Nome  | Idade | Salário |
| ----- | ----- | ------- |
| João  | 25    | 3000    |
| Maria |       | 4500    |
| Pedro | 40    |         |
| Ana   | 22    | 2800    |

Observe:

❌ Idade faltando

❌ Salário faltando

❌ Dados incompletos

Antes de ensinar uma IA precisamos organizar tudo.

É exatamente isso que faremos hoje.

---

# 1. O que é NumPy?

NumPy significa:

```text
Numerical Python
```

É uma biblioteca criada para trabalhar com números e matrizes.

Praticamente todas as bibliotecas de IA utilizam NumPy internamente.

---

## Analogia

Imagine uma calculadora comum.

Ela faz:

```text
2 + 2
```

Agora imagine uma calculadora capaz de realizar milhões de cálculos por segundo.

Essa é a ideia do NumPy.

---

# Importando o NumPy

```python
import numpy as np
```

## Explicação

### import

Importa uma biblioteca.

### numpy

Nome da biblioteca.

### as np

Apelido utilizado mundialmente.

---

# 2. Criando Arrays

Em Machine Learning quase tudo é armazenado em Arrays.

---

## Lista Python

```python
idades = [18, 22, 30, 40]
```

---

## Array NumPy

```python
import numpy as np

idades = np.array([18, 22, 30, 40])

print(idades)
```

Resultado:

```text
[18 22 30 40]
```

---

# O que é um Array?

Um Array é semelhante a uma lista.

Porém:

✅ Mais rápido

✅ Menos memória

✅ Melhor para cálculos matemáticos

---

# 3. Operações Matemáticas

Lista Python:

```python
numeros = [1,2,3]

print(numeros * 2)
```

Resultado:

```text
[1,2,3,1,2,3]
```

A lista foi repetida.

---

Com NumPy:

```python
numeros = np.array([1,2,3])

print(numeros * 2)
```

Resultado:

```text
[2 4 6]
```

Cada elemento foi multiplicado.

---

# Analogia

Imagine uma sala com 40 alunos.

Você deseja aumentar todas as notas em 1 ponto.

Sem NumPy:

```text
Aluno por aluno...
```

Com NumPy:

```Python
import numpy as np

# Notas de 40 alunos
notas = np.array([
    6, 7, 5, 8, 9, 4, 6, 7, 8, 5,
    6, 9, 7, 8, 4, 5, 6, 7, 9, 10,
    5, 6, 7, 8, 9, 4, 6, 7, 8, 5,
    6, 9, 7, 8, 4, 5, 6, 7, 9, 10
])

# Aumentando 1 ponto em todas as notas
notas_corrigidas = notas + 1

print("Notas originais:")
print(notas)

print("Notas com +1 ponto:")
print(notas_corrigidas)

Todos ao mesmo tempo.
```

---

# 4. Matrizes

Machine Learning trabalha principalmente com matrizes.

---

## Criando uma matriz

```python
dados = np.array([
    [18, 1500],
    [25, 3000],
    [35, 5000]
])

print(dados)
```

Resultado:

```text
[[  18 1500]
 [  25 3000]
 [  35 5000]]
```

---

## Visualizando

| Idade | Salário |
| ----- | ------- |
| 18    | 1500    |
| 25    | 3000    |
| 35    | 5000    |

---

# 5. Acessando Posições

```python
dados = np.array([
    [18,1500],
    [25,3000],
    [35,5000]
])

print(dados[0])
```

Resultado:

```text
[18 1500]
```

Primeira linha.

---

## Linha e Coluna

```python
print(dados[1,1])
```

Resultado:

```text
3000
```

---

## Explicação

```text
dados[linha, coluna]
```

---

# 6. Conhecendo Melhor o Pandas

Na aula anterior criamos DataFrames.

Hoje vamos manipulá-los.

---

# Criando uma tabela

```python
import pandas as pd

dados = {
    "nome": ["João", "Maria", "Pedro"],
    "idade": [25, 30, 40]
}

df = pd.DataFrame(dados)

print(df)
```

---

Resultado

| nome  | idade |
| ----- | ----- |
| João  | 25    |
| Maria | 30    |
| Pedro | 40    |

---

# 7. Informações da Tabela

```python
df.info()
```

Resultado:

```text
Quantidade de linhas
Quantidade de colunas
Tipo dos dados
```

---

# Analogia

É como olhar a ficha técnica de um carro.

---

# 8. Estatísticas Básicas

```python
df.describe()
```

Resultado:

```text
Média
Máximo
Mínimo
Desvio padrão
```

---

Exemplo:

| idade        |
| ------------ |
| Média = 31,6 |
| Máximo = 40  |
| Mínimo = 25  |

---

# Por que isso é importante?

Antes de treinar uma IA precisamos conhecer os dados.

---

# 9. Lendo Arquivos CSV

Grande parte dos datasets vêm em CSV.

CSV significa:

```text
Comma Separated Values
```

Valores separados por vírgulas.

---

## Exemplo

arquivo.csv

```csv
nome,idade,salario
João,25,3000
Maria,30,4500
Pedro,40,5000
```

---

# 9.1 Como Carregar Arquivos CSV no Google Colab


```python
df = pd.read_csv("arquivo.csv")
```

Mas para isso funcionar, o arquivo precisa estar disponível dentro do ambiente do Colab.

---

# Opção 1 – Enviar Arquivo do Computador

Esta é a forma mais simples para iniciantes.

Execute:

```python
from google.colab import files

uploaded = files.upload()
```

Ao executar a célula aparecerá um botão:

```text
Escolher arquivos
```

Selecione o arquivo CSV do seu computador.

Exemplo:

```text
alunos.csv
```

Após o upload, o Colab exibirá algo semelhante a:

```text
Saving alunos.csv to alunos.csv
```

Agora o arquivo já está disponível para uso.

---

## Lendo o arquivo

```python
import pandas as pd

df = pd.read_csv("alunos.csv")

print(df)
```
## Em Caso de erro
```python
import pandas as pd

df = pd.read_csv(
    "arquivo.csv",
    sep=";",
    encoding="latin1"
)

print(df)
```
---

# Verificando se o arquivo foi carregado

Você pode listar os arquivos disponíveis:

```python
import os

print(os.listdir())
```

Exemplo de saída:

```text
['alunos.csv']
```

---

# Opção 2 – Utilizando Arquivos do Google Drive

Quando trabalhamos com projetos maiores, é comum armazenar datasets no Google Drive.

---

## Conectando o Drive

```python
from google.colab import drive

drive.mount('/content/drive')
```

Ao executar:

1. Clique no link exibido.
2. Faça login.
3. Copie o código de autorização.
4. Cole no Colab.

---

## Acessando o arquivo

Suponha que o arquivo esteja em:

```text
Meu Drive
└── MachineLearning
    └── alunos.csv
```

O caminho será:

```python
import pandas as pd

df = pd.read_csv(
    "/content/drive/MyDrive/MachineLearning/alunos.csv"
)

print(df)
```

---

# Descobrindo o Caminho do Arquivo

Após montar o Drive, clique no ícone da pasta à esquerda do Colab.

Você verá algo semelhante a:

```text
content
└── drive
    └── MyDrive
```

Navegue até o arquivo.

Clique nos três pontos ao lado do arquivo e escolha:

```text
Copiar caminho
```

Cole diretamente no código:

```python
df = pd.read_csv("CAMINHO_COPIADO")
```

---

# Exercício Prático

Crie uma planilha no Google Planilhas com os dados:

| Nome  | Idade | Nota |
| ----- | ----- | ---- |
| Ana   | 18    | 8    |
| João  | 17    | 7    |
| Maria | 19    | 10   |
| Pedro | 18    | 6    |

---

## Passo 1

Baixe a planilha em formato CSV.

```text
Arquivo → Fazer download → Valores separados por vírgula (.csv)
```

---

## Passo 2

Faça upload para o Colab.

---

## Passo 3

Carregue os dados utilizando:

```python
import pandas as pd

df = pd.read_csv("nome_do_arquivo.csv")

print(df)
```

---

## Passo 4

Mostre as cinco primeiras linhas:

```python
df.head()
```

---

## Passo 5

Mostre informações da tabela:

```python
df.info()
```

---

# Desafio

Após carregar o arquivo, responda:

1. Quantos registros existem?
2. Qual a média das notas?
3. Qual o aluno com a maior nota?
4. Quantos alunos possuem nota maior ou igual a 7?

Dica:

```python
df.describe()
```

e

```python
df[df["Nota"] >= 7]
```

ajudarão a responder essas perguntas.

---

### Observação Importante

No Google Colab, os arquivos enviados pelo upload ficam disponíveis **somente durante a sessão atual**. Se o ambiente for reiniciado, será necessário enviar o arquivo novamente.

Por isso, em projetos maiores, normalmente utilizamos o **Google Drive**, que mantém os arquivos armazenados permanentemente.

---


# 10. Visualizando Dados

Primeiras linhas:

```python
df.head()
```

Resultado:

```text
Mostra as 5 primeiras linhas
```

---

Últimas linhas:

```python
df.tail()
```

Resultado:

```text
Mostra as 5 últimas linhas
```

---

# Analogia

É como abrir uma apostila e olhar a primeira e a última página.

---

# 11. Filtrando Dados

Tabela:

| Nome  | Idade |
| ----- | ----- |
| João  | 18    |
| Ana   | 25    |
| Pedro | 35    |

---

Mostrar somente maiores de idade acima de 20 anos:

```python
print(df[df["idade"] > 20])
```

Resultado:

| Nome  | Idade |
| ----- | ----- |
| Ana   | 25    |
| Pedro | 35    |

---

# Entendendo

```python
df["idade"] > 20
```

Produz:

```text
False
True
True
```

O Pandas utiliza esse resultado para filtrar.

---

# 12. Dados Faltando

Problema muito comum.

---

Tabela:

| Nome  | Idade |
| ----- | ----- |
| João  | 25    |
| Maria |       |
| Pedro | 40    |

---

## Verificando valores vazios

```python
df.isnull()
```

Resultado:

```text
True
False
```

---

## Quantidade de vazios

```python
df.isnull().sum()
```

Exemplo:

```text
idade    1
```

---

# 13. Removendo Dados Vazios

```python
df = df.dropna()
```

---

## Explicação

### drop

Remover

### na

Not Available

(Dado indisponível)

---

# 14. Preenchendo Dados Vazios

Em vez de apagar:

```python
df["idade"] = df["idade"].fillna(df["idade"].mean())
```

---

## O que acontece?

Se a média for:

```text
30
```

Maria recebe:

```text
30
```

automaticamente.

---

# Analogia

Imagine uma pesquisa com 100 pessoas.

Uma delas esqueceu de informar a idade.

Podemos usar a média do grupo para completar.

---

# 15. Preparando Dados para IA

Tabela original:

| Nome  | Idade | Salário | Comprou |
| ----- | ----- | ------- | ------- |
| João  | 25    | 3000    | Sim     |
| Maria | 40    | 5000    | Sim     |
| Pedro | 18    | 1500    | Não     |

---

Separando entradas:

```python
X = df[["idade", "salario"]]
```

---

Separando resposta:

```python
y = df["comprou"]
```

---

Agora os dados estão prontos para treinamento.

---

# Exercício Prático

Crie um DataFrame com:

| Nome  | Nota |
| ----- | ---- |
| Ana   | 8    |
| João  | 5    |
| Pedro | 9    |
| Maria | 7    |

Depois:

1. Mostre a tabela.
2. Exiba apenas alunos com nota maior que 7.
3. Mostre a média das notas.

---

# Exercícios

## Exercício 1

Explique a diferença entre:

```python
lista = [1,2,3]
```

e

```python
np.array([1,2,3])
```

---

## Exercício 2

O que faz:

```python
import numpy as np
```

---

## Exercício 3

Explique:

```python
df.head()
```

---

## Exercício 4

Explique:

```python
df.describe()
```

---

## Exercício 5

Qual a função de:

```python
df.dropna()
```

---

# Desafio da Aula

Crie uma tabela com:

| Horas Estudo | Nota |
| ------------ | ---- |
| 2            | 4    |
| 3            | 5    |
| 5            | 7    |
| 6            | 8    |
| 8            | 10   |

Utilize Pandas para:

* Criar o DataFrame.
* Mostrar estatísticas.
* Exibir apenas notas maiores que 7.

---

# Resumo

Hoje aprendemos:

✅ O que é NumPy

✅ Arrays e matrizes

✅ Operações matemáticas vetorizadas

✅ O que é Pandas

✅ DataFrame

✅ Leitura de CSV

✅ Filtros

✅ Tratamento de dados faltantes

✅ Preparação dos dados para Machine Learning

### Próxima Aula

**Treinamento e Teste de Modelos**

Você aprenderá:

* `train_test_split()`
* Treino e teste
* Overfitting
* Underfitting
* Primeira avaliação de um modelo de Machine Learning com dados reais.
