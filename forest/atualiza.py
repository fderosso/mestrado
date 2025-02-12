from sqlalchemy import create_engine, text

# Configuração da conexão com o PostgreSQL
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'postgres'
POSTGRES_DB = 'postgres'
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = '5432'

# Criar uma conexão com o PostgreSQL
engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')

query = text("""
    SELECT EXTRACT(YEAR FROM data_atendimento) as data_atendimento, bai_id, ate_id
    FROM pesquisa.atendimento
    WHERE /*cid_id = 'F39' and*/ aumentou_atendimento is null
""")

# Executar a consulta e coletar os resultados
with engine.connect() as connection:
    result = connection.execute(query)
    rows = result.fetchall()  # Coletar todos os resultados em uma lista

# Iterar sobre os resultados fora do bloco 'with'

contador = 0
contador2016 = 0
for row in rows:
    data_atendimento = row[0]
    bai_id = row[1]
    if (int(data_atendimento) != 2016):
        with engine.connect() as connection:
            queryVerificaAtual = text(f"SELECT COUNT(*) as atual FROM pesquisa.atendimento WHERE EXTRACT(YEAR FROM data_atendimento) = {int(data_atendimento)} AND bai_id = {bai_id} AND cid_id = 'F39'")
            resultAtual = connection.execute(queryVerificaAtual)
            atual = resultAtual.fetchone()

            queryVerificaAnterior = text(f"SELECT COUNT(*) as atual FROM pesquisa.atendimento WHERE EXTRACT(YEAR FROM data_atendimento) = {int(data_atendimento) -1} AND bai_id = {bai_id} AND cid_id = 'F39'")
            resultAnterior = connection.execute(queryVerificaAnterior)
            anterior = resultAnterior.fetchone()

            atendimentos = '' 
            if (atual[0] > anterior[0]):
                atendimentos = 'Sim'
            elif (atual[0] < anterior[0]):
                atendimentos = 'Não'
            else:
                atendimentos =  'Não'

            with engine.connect() as connection:
                with connection.begin() as transaction:
                    queryUpdate = text(f"UPDATE pesquisa.atendimento SET aumentou_atendimento = '{atendimentos}' WHERE ate_id = {row[2]}")
                    resultado = connection.execute(queryUpdate)
                    contador2016 = contador2016 + 1
        print(str(contador2016)+' Atual: '+str(atual[0])+' Anterior: '+str(anterior[0])+' - '+atendimentos)
        contador = contador + 1
    else:
        with engine.connect() as connection:
                with connection.begin() as transaction:
                    queryUpdate = text(f"UPDATE pesquisa.atendimento SET aumentou_atendimento = 'Não' WHERE ate_id = {row[2]}")
                    resultado = connection.execute(queryUpdate)
print(f'Total: {contador}')