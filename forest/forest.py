import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import plot_tree
from sqlalchemy import create_engine
from itertools import cycle

# Configuração da conexão com o PostgreSQL
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'postgres'
POSTGRES_DB = 'postgres'
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = '5432'

# Criar uma conexão com o PostgreSQL
engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')

# Consulta SQL para selecionar os dados da tabela
query = """
    select 
        a.sexo as sexo, 
        a.energia, 
        d.aba_nome as abastecimento, 
        e.lix_nome as lixo, 
        f.fur_nome as fezes_urina, 
        g.grc_nome as grupo, 
        h.hab_nome as habitacao, 
        i.pro_nome as procedimento,
        j.tdo_nome as tratamento_agua,
        a.aumentou_atendimento
        from 
        pesquisa.atendimento a, 
        pesquisa.abastecimento d,
        pesquisa.destino_lixo e,
        pesquisa.fezes_urina f,
        pesquisa.grupo_comunitario g,
        pesquisa.habitacao h,
        pesquisa.procedimento i,
        pesquisa.tratamento_domicilio j
        where a.tdo_id = j.tdo_id 
        and a.pro_id = i.pro_id 
        and a.hab_id = h.hab_id 
        and a.grc_id = g.grc_id 
        and a.fur_id = f.fur_id 
        and a.lix_id = e.lix_id 
        and a.aba_id = d.aba_id 
        /*and a.cid_id = 'F39'*/
        and a.bai_id <> 76
        /*and a.bai_id = 27*/  
        order by a.ate_id asc
"""

# Carregar os dados do PostgreSQL em um DataFrame do Pandas
df = pd.read_sql(query, con=engine)

# Fechar a conexão com o PostgreSQL
engine.dispose()

# Pesos manuais para cada categoria
weights = {
    'energia': {'Sim': 0.5, 'Nao': 1.0},
    'sexo': {'M': 0.5, 'F': 1.0},
    'abastecimento': {'Rede pública': 2.0, 'Não informado': 0.0, 'Outros': 1.0, 'Poço artesiano': 8.0, 'Carro pipa': 9.0, 'cisterna': 9.0},
    'lixo': {'Coletado': 0.2, 'Não informado': 0.0, 'Queimado / enterrado': 0.8, 'Céu aberto': 1.0, 'Outros': 0.5},
    'fezes_urina': {'Sistema de esgoto': 0.3, 'Fossa séptica': 0.8, 'Não informado': 0.0, 'Céu aberto': 0.9, 'Fossa rudimentar': 0.9, 'Outros': 0.5},
    'grupo': {'Não participa': 0.8, 'Não informado': 0.0, 'Grupo religioso': 0.4, 'Outros': 0.5, 'Associação': 0.6, 'Cooperativa': 0.7},
    'habitacao': {'Tijolo / alvenaria com revestimento': 0.2, 'Madeira emparelhada': 0.4, 'Não informada': 0.0, 'Tijolo / alvenaria sem revestimento': 0.5, 'Outro material': 0.5, 'Taipa sem revestimento': 0.8, 'Material aproveitado': 0.9, 'Taipa com revestimento': 0.7, 'Palha': 0.9},
    'procedimento': {
        'Atendimento médico em unidade de pronto atendimento': 0.9,
        'Consulta / atendimento domiciliar': 0.9,
        'Consulta médica em atenção primária': 0.8,
        'Consulta médica em atenção especializada': 0.9,
        'Atendimento de urgência com observação até 24 horas em atenção especializada': 0.9,
        'Atendimento de urgência em atenção básica': 0.9,
        'Consulta médica em saúde do trabalhador': 0.8,
        'Consulta para acompanhamento de crescimento e desenvolvimento (puericultura)': 0.4,
        'Consulta pré-natal': 0.3,
        'Avaliação antropométrica': 0.2,
        'Consulta em neurologia': 0.7,
        'Consulta puerperal': 0.3,
        'Atendimento / acompanhamento de paciente em reabilitação do desenvolvimento neuropsicomotor': 0.4,
        'Resposta telerregulação / teleconsultoria': 0.7,
        'Atendimento de urgência em atenção primária com remoção': 0.9,
        'Procedimento administrativo': 0.0,
        'Acolhimento com classificação de risco': 0.8,
        'Assistência domiciliar por equipe multiprofissional': 0.8,
        'Teleconsulta médica na atenção especializada': 0.8,
        'Avaliação multidimensional da pessoa idosa': 0.9,
        'Avaliação do desenvolvimento da criança na puericultura': 0.2,
        'Telerregulação / teleconsultoria': 0.6,
        'Parceiro': 0.0,
        'Abordagerm cognitiva comportamental do fumante (por atendimento / paciente)': 0.1,
        'Atendimento / acompanhamento em reabilitação nas múltiplas deficiências': 0.4,
        'Atendimento de urgência em atenção primária com observação até 8 horas': 0.9,
        'Consulta ao paciente curado de tuberculose (tratamento supervisionado)': 0.0,
        'Consulta / atendimentos / acompanhamentos (radar)': 0.8,
        'Teleconsulta na atenção primária': 0.8
    },
    'tratamento_agua': {'Filtração': 0.4, 'Cloração': 0.2, 'Sem tratamento': 0.1, 'Não informado': 0.0, 'Fervura': 0.3},
}

# Função auxiliar para aplicar pesos e lidar com valores None
def apply_weight(column, value):
    if pd.isna(value):
        return 0.0
    if value not in weights[column]:
        return 0.0
    return weights[column][value]

# Aplicar os pesos
df['peso_energia'] = df['energia'].apply(lambda x: apply_weight('energia', x))
df['peso_sexo'] = df['sexo'].apply(lambda x: apply_weight('sexo', x))
df['peso_abastecimento'] = df['abastecimento'].apply(lambda x: apply_weight('abastecimento', x))
df['peso_lixo'] = df['lixo'].apply(lambda x: apply_weight('lixo', x))
df['peso_fezes_urina'] = df['fezes_urina'].apply(lambda x: apply_weight('fezes_urina', x))
df['peso_grupo'] = df['grupo'].apply(lambda x: apply_weight('grupo', x))
df['peso_habitacao'] = df['habitacao'].apply(lambda x: apply_weight('habitacao', x))
df['peso_procedimento'] = df['procedimento'].apply(lambda x: apply_weight('procedimento', x))
df['peso_tratamento'] = df['tratamento_agua'].apply(lambda x: apply_weight('tratamento_agua', x))

# Garantir que a coluna de destino tem valores apropriados
df['aumentou_atendimento'] = df['aumentou_atendimento'].map({'Sim': 1, 'Não': 0})

# Preparar os dados ponderados para treinamento
X_weighted = df[['peso_energia', 'peso_sexo', 'peso_abastecimento', 'peso_lixo', 'peso_fezes_urina', 'peso_grupo', 'peso_habitacao', 'peso_procedimento', 'peso_tratamento']]

y = df['aumentou_atendimento']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=61658)

# Criar e treinar o modelo RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1000, random_state=61658)
rf_model.fit(X_train, y_train)

# Prever com o conjunto de teste
y_pred = rf_model.predict(X_test)

# Avaliação do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("relatório de classificação:\n", classification_report(y_test, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

# Gráfico de Precisão
accuracy = accuracy_score(y_test, y_pred)
labels = ['Predições corretas', 'Predições incorretas']
values = [accuracy, 1 - accuracy]

plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['green', 'red'])
plt.title('Acurácia de Classificação Random Forest')
plt.ylabel('Proportion')
plt.show()

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predição não aumentou', 'Predição aumentou'], yticklabels=['Atual não aumentou', 'Atual aumentou'])
plt.title('Matriz de confusão')
plt.xlabel('Predição')
plt.ylabel('Atual')
plt.show()


# Importância das variáveis
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotar
plt.figure(figsize=(10, 6))
plt.title("Importância das Variáveis")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
plt.xlabel('Variável')
plt.ylabel('Importância')
plt.show()

# Comparar predições com rótulos verdadeiros
results = pd.DataFrame({'Verdadeiro': y_test, 'Predito': y_pred})

# Encontrar predições incorretas
incorrect = results[results['Verdadeiro'] != results['Predito']]

# Plotar predições incorretas
plt.figure(figsize=(10, 6))
plt.scatter(range(len(incorrect)), incorrect['Verdadeiro'], color='red', label='Verdadeiro')
plt.scatter(range(len(incorrect)), incorrect['Predito'], color='blue', label='Predito')
plt.legend()
plt.xlabel('Instância')
plt.ylabel('Classe')
plt.title('Análise de Erros')
plt.show()

# Curva ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]
y_score = rf_model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Classe {0} (área = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para cada classe')
plt.legend(loc="lower right")
plt.show()