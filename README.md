# Interface de execução do algoritmo Random Forest e geração de gráficos
## Estrutura de pastas

O estudo é dividido em 3 pastas:

1. pasta `dados`:

   - possui o dump da base de dados PostgreSQL.
   - esta base contempla os dados de atendimentos do ano de 2016 até junho de 2024.

2. pasta `forest`:

   - possui os códigos em linguagem python.
   - o arquivo `main.py` é o arquivo inicial.

2. pasta `auxiliar`:

   - contém arquivos que foram utilizados para testes e limpeza dos dados.

## Getting started

1. criar localmente o arquivo `.env` com variáveis para conexão com o banco de dados, este arquivo deve estar na mesma pasta dos arquivos da pasta forest.
2. instalar o banco de dados PostgreSQL localmente.
3. rodar o dump da base de dados que consta na pasta dados.
4. instalar as dependências de python através do comando `pip install -r requirements.txt` dentro da pasta forest.
5. executar o arquivo `main.py`.
