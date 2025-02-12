import sys
import json
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QLineEdit, QComboBox, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from PyQt5.QtGui import QDoubleValidator
from unidecode import unidecode
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
from itertools import cycle
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
import pandas as pd

class Energia:
    def __init__(self):
        self.values = {}

class Sexo:
    def __init__(self):
        self.values = {}

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, data):
        self.axes.clear()
        self.axes.plot(data)
        self.draw()

class MainWindow(QMainWindow):
    bairroTitulo = ""
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Janela com Abas e Gráficos")
        self.resize(1500, 600)
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
            self.itemsEnergia = [('Sim', '1.0'), ('Nao', '5.0')]
            self.itemsGenero = [('Masculino', '2.0'), ('Feminino', '5.0')]

            self.combo_data_bairro = dfComboDataBairro.values.tolist()
            self.combo_data_cid = dfComboDataCID.values.tolist()

        except Exception as e:
            print(f"Ocorreu um erro ao carregar dados auxiliares: {e}")
        finally:        
            # Fechar a conexão com o PostgreSQL
            engine.dispose()
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.energia = Energia()
        self.sexo = Sexo()
        self.line_edits = {}  # Inicializa o dicionário aqui
        self.comboboxes = {}
        self.mappings = {}    # Armazena o mapeamento identificador -> valor textual
        
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        self.tab6 = QWidget()
        
        self.tabs.addTab(self.tab1, "Preenchimento de variáveis")
        self.tabs.addTab(self.tab2, "Acurácia de Classificação Random Forest")
        self.tabs.addTab(self.tab3, "Matriz de confusão")
        self.tabs.addTab(self.tab4, "Curva ROC para cada classe")
        self.tabs.addTab(self.tab5, "Importância das variáveis")
        self.tabs.addTab(self.tab6, "Análise de erros")
        
        self.init_tab1()
        self.init_tab_graph(self.tab2)
        self.init_tab_graph(self.tab3)
        self.init_tab_graph(self.tab4)
        self.init_tab_graph(self.tab5)
        self.init_tab_graph(self.tab6)
        
    def init_tab1(self):
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
        self.create_variable_group(self.itemsGenero, 'Sexo', main_layout_1)
        self.create_variable_group(self.itemsGruposComunitarios, 'Grupos comunitários', main_layout_1)
        self.create_variable_group(self.itemsAbastecimento, 'Abastecimento', main_layout_1)
        self.create_variable_group(self.itemsLixo, 'Destinação do lixo', main_layout_1)
      
        # Segunda coluna
        self.create_variable_group(self.itemsHabitacao, 'Tipo de habitação', main_layout_2)
        self.create_variable_group(self.itemsTratamentoSanitario, 'Tratamento sanitário', main_layout_2)
        self.create_variable_group(self.itemsSaneamento, 'Saneamento básico', main_layout_2)
        self.create_variable_group(self.itemsEnergia, 'Energia elétrica', main_layout_2)
        
        hbox_sim_params_1 = QHBoxLayout()
        label_test_size = QLabel("Tamanho do teste:", self)
        self.input_test_size = QLineEdit(self)
        self.input_test_size.setFixedSize(50, 20)

        label_random_state = QLabel("Estado randomico:", self)
        self.input_random_state = QLineEdit(self)
        self.input_random_state.setFixedSize(50, 20)

        hbox_sim_params_1.addWidget(label_test_size)
        hbox_sim_params_1.addWidget(self.input_test_size)
        hbox_sim_params_1.addWidget(label_random_state)
        hbox_sim_params_1.addWidget(self.input_random_state)

        # Layout Horizontal para os próximos dois campos (estimadores e profundidade)
        hbox_sim_params_2 = QHBoxLayout()
        label_n_estimators = QLabel("Estimadores:", self)
        self.input_n_estimators = QLineEdit(self)
        self.input_n_estimators.setFixedSize(50, 20)

        label_max_depth = QLabel("Profundidade:", self)
        self.input_max_depth = QLineEdit(self)
        self.input_max_depth.setFixedSize(50, 20)

        hbox_sim_params_2.addWidget(label_n_estimators)
        hbox_sim_params_2.addWidget(self.input_n_estimators)
        hbox_sim_params_2.addWidget(label_max_depth)
        hbox_sim_params_2.addWidget(self.input_max_depth)

        # Layout Vertical Principal para combinar os dois layouts horizontais
        vbox_sim_params = QVBoxLayout()
        vbox_sim_params.addLayout(hbox_sim_params_1)
        vbox_sim_params.addLayout(hbox_sim_params_2)

        # Adicionar ao layout principal, acima do botão 'Executar simulação'
        self.main_layout_3.addLayout(vbox_sim_params)

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
        self.tab1.setLayout(hbox_main)
        
    def init_tab_graph(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

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

            v_test_size = float(self.input_test_size.text()) if self.input_test_size.text() else 0.2
            v_random_state = int(self.input_random_state.text()) if self.input_random_state.text() else 61658
            v_n_estimators = int(self.input_n_estimators.text()) if self.input_n_estimators.text() else 100
            v_max_depth = int(self.input_max_depth.text()) if self.input_max_depth.text() else 100

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
                        bairroTitulo = combobox_values[title]
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
                msg.setIcon(QMessageBox.Critical)
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
            X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=v_test_size, random_state=v_random_state)

            # Criar e treinar o modelo RandomForestClassifier
            rf_model = RandomForestClassifier(n_estimators=v_n_estimators, random_state=v_random_state, max_depth=v_max_depth)
            rf_model.fit(X_train, y_train)

            # Prever com o conjunto de teste
            y_pred = rf_model.predict(X_test)

            # Avaliação do modelo
            print("Acurácia:", accuracy_score(y_test, y_pred))
            print("Relatório de classificação:\n", classification_report(y_test, y_pred))
            print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

            self.criaGraficoColuna3(self.tab2, y_test, y_pred, rf_model, X_test, y, X_train)
            self.criaGraficoColuna4(self.tab3, y_test, y_pred, rf_model, X_test, y, X_train)

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Simulação executada.")
            msg.setWindowTitle("Aviso")
            msg.setStyleSheet("""
            QMessageBox {
                border: 1px solid black;
                border-radius: 10px;
                min-width: 700px;
            }
        """)
            self_center = self.frameGeometry()
            self_center.moveCenter(self.geometry().center())
            msg.move(self_center.center())
            msg.exec_()
            return



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
        combobox.objectName = "comboBairro"
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

    def criaGraficoColuna3(self, layout, y_test, y_pred, rf_model, X_test, y, X_train):

        for i in reversed(range(self.tab2.layout().count())):
            widget_to_remove = self.tab2.layout().itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        for i in reversed(range(self.tab3.layout().count())):
            widget_to_remove = self.tab3.layout().itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        for i in reversed(range(self.tab4.layout().count())):
            widget_to_remove = self.tab4.layout().itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        for i in reversed(range(self.tab5.layout().count())):
            widget_to_remove = self.tab5.layout().itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        for i in reversed(range(self.tab6.layout().count())):
            widget_to_remove = self.tab6.layout().itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        # Gráfico de Precisão
        accuracy = accuracy_score(y_test, y_pred)
        labels = ['Predições corretas', 'Predições incorretas']
        values = [accuracy, 1 - accuracy]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=['green', 'red'])
        #plt.title(f"Acurácia")
        plt.ylabel('Proporção', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        canvasGrafico = FigureCanvas(plt.gcf())

        self.tab2.layout().addWidget(canvasGrafico)

        plt.close()

        # Matriz de Confusão
        conf_matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 18}, fmt='d', cmap='Blues', xticklabels=['Predição não aumentou', 'Predição aumentou'], yticklabels=['Atual não aumentou', 'Atual aumentou'])
        plt.xlabel('Predição', fontsize=18)
        plt.ylabel('Atual', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        canvasGrafico = FigureCanvas(plt.gcf())
        #layout.addWidget(canvasGrafico)
        self.tab3.layout().addWidget(canvasGrafico)
        plt.close()
 
    def criaGraficoColuna4(self, layout, y_test, y_pred, rf_model, X_test, y, X_train):
        
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
        colors = cycle(['blue', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Classe {0} (área = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos',fontsize=18)
        plt.ylabel('Taxa de Verdadeiros Positivos',fontsize=18)
        #plt.title('Curva ROC para cada classe')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.rc('legend', fontsize=18)
        plt.legend(loc="lower right")
        canvasGrafico = FigureCanvas(plt.gcf())
        #layout.addWidget(canvasGrafico)
        self.tab4.layout().addWidget(canvasGrafico)
        plt.close()

        # Importância das variáveis
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plotar
        variaveis = ['Sexo','Fezes / urina','Tratamento','Energia','Lixo','Abastecimento','Grupo','Habitação']
        plt.figure(figsize=(10, 6))
        #plt.title("Importância das Variáveis")
        plt.barh(range(X_train.shape[1]), importances[indices], align="center")
        plt.yticks(range(X_train.shape[1]), [X_train.columns[i][2:] for i in indices])
        plt.xlabel('Importância',fontsize=18)
        plt.ylabel('Variável',fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=16)

        canvasGrafico = FigureCanvas(plt.gcf())
        #layout.addWidget(canvasGrafico)
        self.tab5.layout().addWidget(canvasGrafico)
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
        plt.xlabel('Instância',fontsize=18)
        plt.ylabel('Classe')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #plt.title('Análise de Erros')
        canvasGrafico = FigureCanvas(plt.gcf())
        #layout.addWidget(canvasGrafico)
        self.tab6.layout().addWidget(canvasGrafico)
        plt.close()

        
    def show_graphs(self):
        # Gerar dados dummy para os gráficos
        data1 = [1, 2, 3, 4, 5]
        data2 = [5, 4, 3, 2, 1]
        data3 = [2, 3, 2, 3, 2]
        data4 = [1, 3, 2, 4, 5]
        
        # Criar gráficos em cada aba correspondente
        self.plot_graph(self.tab2, data1, "Gráfico 1")
        self.plot_graph(self.tab3, data2, "Gráfico 2")
        self.plot_graph(self.tab4, data3, "Gráfico 3")
        self.plot_graph(self.tab5, data4, "Gráfico 4")
        self.plot_graph(self.tab6, data4, "Gráfico 5")
        
    def plot_graph(self, tab, data, title):
        # Remove o label de "Gráfico não carregado" se existir
        if isinstance(tab.layout().itemAt(0).widget(), QLabel):
            tab.layout().itemAt(0).widget().deleteLater()
        
        # Cria o canvas para o gráfico
        canvas = GraphCanvas(tab)
        canvas.plot(data)
        tab.layout().addWidget(canvas)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
