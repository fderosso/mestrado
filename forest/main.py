import json
import sys
import os
import pandas as pd
from PyQt5.QtGui import QDoubleValidator
import matplotlib.pyplot as plt
from unidecode import unidecode
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QComboBox, QMessageBox
from sqlalchemy import create_engine, text
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
from itertools import cycle

class Energia:
    def __init__(self):
        self.values = {}

class Sexo:
    def __init__(self):
        self.values = {}

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.energia = Energia()
        self.sexo = Sexo()
        self.line_edits = {}  # Inicializa o dicionário aqui
        self.comboboxes = {}
        self.mappings = {}    # Armazena o mapeamento identificador -> valor textual
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Simulação usando Random Forest')
        self.setGeometry(50, 50, 2200, 900)
        load_dotenv()

        # Criar uma conexão com o PostgreSQL
        try:
            engine = create_engine(f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}')
        except Exception as e:
            print(f"Ocorreu um erro ao criar conexão com tabelas auxiliares: {e}")

        # Dados para as listas(para cada array fazer um select na tabela)
        # Consulta SQL para selecionar os dados da tabela
        queryAbastecimento = """ select aba_nome as abastecimento, peso::text from pesquisa.abastecimento """
        queryLixo = """ select lix_nome as lixo, peso::text from pesquisa.destino_lixo """
        queryFezesUrina = """ select fur_nome as fezes_urina, peso::text from pesquisa.fezes_urina """
        queryGrupoComunitario = """ select grc_nome as grupo_comunitario, peso::text from pesquisa.grupo_comunitario """
        queryHabitacao = """ select hab_nome as habitacao, peso::text from pesquisa.habitacao """
        queryTratamentoSanitario = """ select tdo_nome as tratamento_sanitario, peso::text from pesquisa.tratamento_domicilio """
        queryBairro = """ select bai_nome as display_name, bai_id::text as value from pesquisa.bairro order by display_name """
        queryCID = """ select distinct b.cid_nome as display_name, b.cid_id as value from pesquisa.atendimento a, pesquisa.cid b where a.cid_id = b.cid_id order by display_name """
        
        # Carregar os dados do PostgreSQL em um DataFrame do Pandas
        try:
            dfAbastecimento = pd.read_sql(queryAbastecimento, con=engine)
            dfLixo = pd.read_sql(queryLixo, con=engine)
            dfFezesUrina = pd.read_sql(queryFezesUrina, con=engine)
            dfGrupoComunitario = pd.read_sql(queryGrupoComunitario, con=engine)
            dfHabitacao = pd.read_sql(queryHabitacao, con=engine)
            dfTratamentoSanitario = pd.read_sql(queryTratamentoSanitario, con=engine)
            dfComboDataBairro = pd.read_sql(queryBairro, con=engine)
            dfComboDataCID = pd.read_sql(queryCID, con=engine)

            self.itemsGruposComunitarios = dfGrupoComunitario.values
            self.itemsAbastecimento = dfAbastecimento.values
            self.itemsLixo = dfLixo.values
            self.itemsHabitacao = dfHabitacao.values
            self.itemsTratamentoSanitario = dfTratamentoSanitario.values
            self.itemsSaneamento = dfFezesUrina.values
            self.itemsEnergia = [('Sim', '0.5'), ('Nao', '1.0')]
            self.itemsGenero = [('Masculino', '0.5'), ('Feminino', '1.0')]

            self.combo_data_bairro = dfComboDataBairro.values.tolist()
            self.combo_data_cid = dfComboDataCID.values.tolist()

        except Exception as e:
            print(f"Ocorreu um erro ao carregar dados auxiliares: {e}")
        finally:        
            # Fechar a conexão com o PostgreSQL
            engine.dispose()
            
        # Layout Principal
        main_layout_1 = QVBoxLayout()
        main_layout_1.setContentsMargins(10, 10, 10, 10)
        main_layout_1.setSpacing(10)
        main_layout_2 = QVBoxLayout()
        main_layout_2.setContentsMargins(10, 10, 10, 10)
        main_layout_2.setSpacing(10)
        self.main_layout_3 = QVBoxLayout()
        self.main_layout_3.setContentsMargins(10, 10, 10, 10)
        self.main_layout_3.setSpacing(10)

        self.main_layout_4 = QVBoxLayout()
        self.main_layout_4.setContentsMargins(10, 10, 10, 10)
        self.main_layout_4.setSpacing(10)

        # Primeira coluna
        self.add_combobox_from_query('Bairro: ', self.combo_data_bairro, main_layout_1)
        self.add_combobox_from_query('CID: ', self.combo_data_cid, main_layout_1)
        #self.add_combobox('CID: ', ['Opção 1', 'Opção 2', 'Opção 3'], main_layout_1)
        self.create_variable_group(self.itemsGenero, 'Sexo', main_layout_1)
        self.create_variable_group(self.itemsGruposComunitarios, 'Grupos comunitários', main_layout_1)
        self.create_variable_group(self.itemsAbastecimento, 'Abastecimento', main_layout_1)
        self.create_variable_group(self.itemsLixo, 'Destinação do lixo', main_layout_1)
      
        # Segunda coluna
        self.create_variable_group(self.itemsHabitacao, 'Tipo de habitação', main_layout_2)
        self.create_variable_group(self.itemsTratamentoSanitario, 'Tratamento sanitário', main_layout_2)
        self.create_variable_group(self.itemsSaneamento, 'Saneamento básico', main_layout_2)
        self.create_variable_group(self.itemsEnergia, 'Energia elétrica', main_layout_2)
        self.right_button = QPushButton('Executar simulação', self)
        self.right_button.clicked.connect(self.get_values)
        main_layout_2.addWidget(self.right_button)

        # Layout Horizontal Principal
        hbox_main = QHBoxLayout()
        hbox_main.addLayout(main_layout_1)
        hbox_main.addLayout(main_layout_2)
        hbox_main.addLayout(self.main_layout_3)
        hbox_main.addLayout(self.main_layout_4)

        # Configurando o Layout da Janela
        self.setLayout(hbox_main)

    def add_record(self, name, value, identifier):
        hbox = QHBoxLayout()
        hbox.setSpacing(5)
        label = QLabel(name, self)
        label.setObjectName("lbl" + identifier)
        line_edit = QLineEdit(value, self)
        line_edit.setFixedSize(50, 20)
        line_edit.setObjectName("edit" + identifier)

        # Guardar a referência do QLineEdit no dicionário
        self.line_edits[identifier] = line_edit
        self.mappings[identifier] = name  # Armazena o mapeamento identificador -> valor textual

        # Configura o validador para aceitar apenas valores float
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setDecimals(2)
        line_edit.setValidator(validator)

        hbox.addWidget(label)
        hbox.addWidget(line_edit)
        self.vbox_list.addLayout(hbox)

    def add_combobox_from_query(self, title, items, layout):
        hbox = QHBoxLayout()
        hbox.setSpacing(5)
        label = QLabel(title, self)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
                
        combobox = QComboBox(self)
        combobox.setFixedSize(500, 30)
        combobox.addItem('Todos', '0')
        for item in items:
            if isinstance(item, list) and len(item) == 2:
                display_name, value = item
                combobox.addItem(display_name, value)  # Adiciona item com valor associado
            else:
                print(f"Item inválido: {item}")
        
        combobox.setObjectName("combo" + title)

        # Guardar a referência do QComboBox no dicionário
        self.comboboxes[title] = combobox

        hbox.addWidget(label)
        hbox.addWidget(combobox)

        layout.addLayout(hbox)

    def create_variable_group(self, values, title, layout):
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Layout Vertical para os Registros
        self.vbox_list = QVBoxLayout()

        # Adiciona cada item ao layout
        for index, (display_name, value) in enumerate(values):
            identifier = f"{unidecode(title.replace(' ', '_'))}_{index}"
            self.add_record(display_name, value, identifier)
        layout.addLayout(self.vbox_list)

    def get_values(self):
            # Cria um dicionário para armazenar os resultados por categoria
            energia_values = {}
            sexo_values = {}
            grupos_comunitarios_values = {}
            abastecimento_values = {}
            destinacao_do_lixo_values = {}
            tipo_de_habitacao_values = {}
            saneamento_basico_values = {}
            tratamento_sanitario_values = {}

            combobox_values = {}

            for identifier, line_edit in self.line_edits.items():
                value = line_edit.text()
                display_value = self.mappings.get(identifier, "Unknown")
                try:
                    float_value = float(value)
                except ValueError:
                    continue

                # Separar os valores por categoria
                if 'Energia' in identifier:
                    energia_values[display_value] = float_value
                elif 'Sexo' in identifier:
                    sexo_values[display_value] = float_value
                elif 'Grupos_comunitarios' in identifier:
                    grupos_comunitarios_values[display_value] = float_value    
                elif 'Abastecimento' in identifier:
                    abastecimento_values[display_value] = float_value    
                elif 'Destinacao_do_lixo' in identifier:
                    destinacao_do_lixo_values[display_value] = float_value    
                elif 'Tipo_de_habitacao' in identifier:
                    tipo_de_habitacao_values[display_value] = float_value    
                elif 'Saneamento_basico' in identifier:
                    saneamento_basico_values[display_value] = float_value    
                elif 'Tratamento_sanitario' in identifier:
                    tratamento_sanitario_values[display_value] = float_value

            # Inicializar variáveis cidId e baiId
            cidId = ""
            baiId = ""

            for title, combobox in self.comboboxes.items():
                selected_value = combobox.currentData()  # Obtém o valor selecionado do QComboBox
                combobox_values[title] = selected_value
                if title == 'Bairro: ':
                    if combobox_values[title] == '0':
                        baiId = ""
                    else:
                        baiId = f"and a.bai_id = {combobox_values[title]}"
                elif title == 'CID: ':
                    if combobox_values[title] == '0':
                        cidId = ""
                    else:
                        cidId = f"and a.cid_id = '{combobox_values[title]}'"  

            weights = {
                'energia': energia_values,
                'sexo': sexo_values,
                'abastecimento': abastecimento_values,
                'lixo': destinacao_do_lixo_values,
                'fezes_urina': tratamento_sanitario_values,
                'grupo': grupos_comunitarios_values,
                'habitacao': tipo_de_habitacao_values,
                'tratamento_agua': saneamento_basico_values,
            }

            objeto_str = str(weights)
            with open('saida.txt', 'w') as file:
                json.dump(objeto_str, file, indent=4)

            load_dotenv()
            try:
                engine = create_engine(f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}')
            except Exception as e:
                print(f"Ocorreu um erro ao carregar dados de atendimentos: {e}")

            query = f"""
                select 
                    a.sexo as sexo, a.energia, d.aba_nome as abastecimento, 
                    e.lix_nome as lixo, f.fur_nome as fezes_urina, 
                    g.grc_nome as grupo, h.hab_nome as habitacao, 
                    j.tdo_nome as tratamento_agua, a.aumentou_atendimento
                    from pesquisa.atendimento a
                    join pesquisa.abastecimento d on a.aba_id = d.aba_id
                    join pesquisa.destino_lixo e on a.lix_id = e.lix_id
                    join pesquisa.fezes_urina f on a.fur_id = f.fur_id
                    join pesquisa.grupo_comunitario g on a.grc_id = g.grc_id
                    join pesquisa.habitacao h on a.hab_id = h.hab_id
                    join pesquisa.tratamento_domicilio j on a.tdo_id = j.tdo_id
                    join pesquisa.bairro k on a.bai_id = k.bai_id
                    where 1=1
                    {cidId}
                    {baiId}
                    order by a.ate_id asc
            """
            #print(query)
            df = None
            try:
                df = pd.read_sql(query, con=engine)
            except Exception as e:
                print(f"Ocorreu um erro ao retornar dados de atendimentos: {e}")
            finally:        
                # Fechar a conexão com o PostgreSQL
                engine.dispose()  
            
            if df is None:
                print("DataFrame is None. Check the query or connection to the database.")
                return

            if df.empty:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("Nenhum registro encontrado.")
                msg.setWindowTitle("Aviso")
                msg.setStyleSheet("""
            QMessageBox {
                border: 2px solid black;
                border-radius: 10px;
            }
        """)
                self_center = self.frameGeometry()
                self_center.moveCenter(self.geometry().center())
                msg.move(self_center.center())
                msg.exec_()
                return
            
            # Função auxiliar para aplicar pesos e lidar com valores None
            def apply_weight(column, value):
                if pd.isna(value):
                    return 0.0
                if value not in weights[column]:
                    return 0.0
                return weights[column][value]
            
            # Aplicar os pesos
            df['p_energia'] = df['energia'].apply(lambda x: apply_weight('energia', x))
            df['p_sexo'] = df['sexo'].apply(lambda x: apply_weight('sexo', x))
            df['p_abastecimento'] = df['abastecimento'].apply(lambda x: apply_weight('abastecimento', x))
            df['p_lixo'] = df['lixo'].apply(lambda x: apply_weight('lixo', x))
            df['p_fezes_urina'] = df['fezes_urina'].apply(lambda x: apply_weight('fezes_urina', x))
            df['p_grupo'] = df['grupo'].apply(lambda x: apply_weight('grupo', x))
            df['p_habitacao'] = df['habitacao'].apply(lambda x: apply_weight('habitacao', x))
            df['p_tratamento'] = df['tratamento_agua'].apply(lambda x: apply_weight('tratamento_agua', x))

            # Garantir que a coluna p_sexo tem valores apropriados
            df['p_sexo'] = df['p_sexo'].map({'Masculino': 'M', 'Feminino': 'F'})

            # Garantir que a coluna de aumentou_atendimento tem valores apropriados
            df['aumentou_atendimento'] = df['aumentou_atendimento'].map({'Sim': 1, 'Não': 0})

            # Preparar os dados ponderados para treinamento
            X_weighted = df[['p_energia', 'p_sexo', 'p_abastecimento', 'p_lixo', 'p_fezes_urina', 'p_grupo', 'p_habitacao', 'p_tratamento']]

            y = df['aumentou_atendimento']

            # Dividir os dados em conjunto de treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=61658)

            # Criar e treinar o modelo RandomForestClassifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=61658, max_depth=100)
            rf_model.fit(X_train, y_train)

            # Prever com o conjunto de teste
            y_pred = rf_model.predict(X_test)

            # Avaliação do modelo
            print("Acurácia:", accuracy_score(y_test, y_pred))
            print("Relatório de classificação:\n", classification_report(y_test, y_pred))
            print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

            self.criaGraficoColuna3(self.main_layout_3, y_test, y_pred, rf_model, X_test, y, X_train)
            self.criaGraficoColuna4(self.main_layout_4, y_test, y_pred, rf_model, X_test, y, X_train)


    def criaGraficoColuna3(self, layout, y_test, y_pred, rf_model, X_test, y, X_train):
        # Remove o widget existente, se houver
        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        # Gráfico de Precisão
        accuracy = accuracy_score(y_test, y_pred)
        labels = ['Predições corretas', 'Predições incorretas']
        values = [accuracy, 1 - accuracy]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, values, color=['green', 'red'])
        plt.title('Acurácia de Classificação Random Forest')
        plt.ylabel('Proporção')

        canvasGrafico = FigureCanvas(plt.gcf())
        layout.addWidget(canvasGrafico)
        plt.close()

        # Matriz de Confusão
        conf_matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predição não aumentou', 'Predição aumentou'], yticklabels=['Atual não aumentou', 'Atual aumentou'])
        plt.title('Matriz de confusão')
        plt.xlabel('Predição')
        plt.ylabel('Atual')
        canvasGrafico = FigureCanvas(plt.gcf())
        layout.addWidget(canvasGrafico)
        plt.close()

    def criaGraficoColuna4(self, layout, y_test, y_pred, rf_model, X_test, y, X_train):
        # Remove o widget existente, se houver
        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

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
        canvasGrafico = FigureCanvas(plt.gcf())
        layout.addWidget(canvasGrafico)
        plt.close()

        # Importância das variáveis
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plotar
        variaveis = ['Sexo','Fezes / urina','Tratamento','Energia','Abastecimento','Lixo','Grupo','Habitação']
        plt.figure(figsize=(10, 6))
        plt.title("Importância das Variáveis")
        plt.barh(range(X_train.shape[1]), importances[indices], align="center")
        plt.yticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices])
        plt.xlabel('Importância')
        plt.ylabel('Variável')

        canvasGrafico = FigureCanvas(plt.gcf())
        layout.addWidget(canvasGrafico)
        plt.close()

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
        canvasGrafico = FigureCanvas(plt.gcf())
        layout.addWidget(canvasGrafico)
        plt.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())