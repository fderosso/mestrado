import sys
import pandas as pd
from PyQt5.QtGui import QDoubleValidator
import matplotlib.pyplot as plt
from unidecode import unidecode
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QGroupBox
from sqlalchemy import create_engine

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Simulação usando Random Forest')
        self.setGeometry(100, 100, 1600, 800)

        # Configuração da conexão com o PostgreSQL
        POSTGRES_USER = 'postgres'
        POSTGRES_PASSWORD = 'postgres'
        POSTGRES_DB = 'postgres'
        POSTGRES_HOST = 'localhost'
        POSTGRES_PORT = '5432'

        # Criar uma conexão com o PostgreSQL
        engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')

        # Dados para as listas(para cada array fazer um select na tabela)
        # Consulta SQL para selecionar os dados da tabela
        queryAbastecimento = """ select aba_nome as abastecimento, peso::text from pesquisa.abastecimento """
        queryLixo = """ select lix_nome as lixo, peso::text from pesquisa.destino_lixo """
        queryFezesUrina = """ select fur_nome as fezes_urina, peso::text from pesquisa.fezes_urina """
        queryGrupoComunitario = """ select grc_nome as grupo_comunitario, peso::text from pesquisa.grupo_comunitario """
        queryHabitacao = """ select hab_nome as habitacao, peso::text from pesquisa.habitacao """
        queryTratamentoSanitario = """ select tdo_nome as tratamento_sanitario, peso::text from pesquisa.tratamento_domicilio """
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
        dfAbastecimento = pd.read_sql(queryAbastecimento, con=engine)
        dfLixo = pd.read_sql(queryLixo, con=engine)
        dfFezesUrina = pd.read_sql(queryFezesUrina, con=engine)
        dfGrupoComunitario = pd.read_sql(queryGrupoComunitario, con=engine)
        dfHabitacao = pd.read_sql(queryHabitacao, con=engine)
        dfTratamentoSanitario = pd.read_sql(queryTratamentoSanitario, con=engine)
        df = pd.read_sql(query, con=engine)
        # Fechar a conexão com o PostgreSQL
        engine.dispose()

        self.itemsGruposComunitarios = dfGrupoComunitario.values
        self.itemsAbastecimento = dfAbastecimento.values
        self.itemsLixo = dfLixo.values
        self.itemsHabitacao = dfHabitacao.values
        self.itemsTratamentoSanitario = dfTratamentoSanitario.values
        self.itemsSaneamento = dfFezesUrina.values
        self.itemsEnergia = [('Com energia', '0.5'),('Sem energia', '1.0'),]
        self.itemsGenero = [('Masculino', '0.5'),('Feminino', '1.0'),]

        # Layout Principal
        main_layout_1 = QVBoxLayout()
        main_layout_1.setContentsMargins(10, 10, 10, 10)
        main_layout_1.setSpacing(10)
        main_layout_2 = QVBoxLayout()
        main_layout_2.setContentsMargins(10, 10, 10, 10)
        main_layout_2.setSpacing(10)
        main_layout_3 = QVBoxLayout()
        main_layout_3.setContentsMargins(10, 10, 10, 10)
        main_layout_3.setSpacing(10)

        # Primeira coluna
        self.create_variable_group(self.itemsGenero, 'Gênero', main_layout_1)
        self.create_variable_group(self.itemsGruposComunitarios, 'Grupos comunitários', main_layout_1)
        self.create_variable_group(self.itemsAbastecimento, 'Abastecimento', main_layout_1)
        self.create_variable_group(self.itemsLixo, 'Destinação do lixo', main_layout_1)

        # Segunda coluna
        self.create_variable_group(self.itemsHabitacao, 'Tipo de habitação', main_layout_2)
        self.create_variable_group(self.itemsTratamentoSanitario, 'Tratamento sanitário', main_layout_2)
        self.create_variable_group(self.itemsSaneamento, 'Saneamento básico', main_layout_2)
        self.create_variable_group(self.itemsEnergia, 'Energia elétrica', main_layout_2)
        self.right_button = QPushButton('Executar simulação', self)
        self.right_button.clicked.connect(self.on_button_click)
        main_layout_2.addWidget(self.right_button)

        # Terceira coluna
        self.create_example_plot(main_layout_3)

        # Layout Horizontal Principal
        hbox_main = QHBoxLayout()
        hbox_main.addLayout(main_layout_1)
        hbox_main.addLayout(main_layout_2)
        hbox_main.addLayout(main_layout_3)

        # Configurando o Layout da Janela
        self.setLayout(hbox_main)

    def add_record(self, name, value, identifier):
        hbox = QHBoxLayout()
        hbox.setSpacing(5)
        label = QLabel(name, self)
        label.setObjectName("lbl"+identifier)
        line_edit = QLineEdit(value, self)
        line_edit.setFixedSize(50, 20)
        line_edit.setObjectName("edit"+identifier)


        # Configura o validador para aceitar apenas valores float
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setDecimals(2)
        line_edit.setValidator(validator)

        hbox.addWidget(label)
        hbox.addWidget(line_edit)
        self.vbox_list.addLayout(hbox)

    def create_variable_group(self, values, title, layout):
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Layout Vertical para os Registros
        self.vbox_list = QVBoxLayout()
        self.vbox_list.setSpacing(4)

        # Adicionando os Registros
        for index, (name, value) in enumerate(values):
            sanitized_title = unidecode(title.replace(' ', '_'))
            identifier = f"{sanitized_title}_{index}"
            print(identifier.lower())
            self.add_record(name, value, identifier)

        # Grupo para Moldura
        group_box = QGroupBox()
        group_box.setLayout(self.vbox_list)
        group_box.setStyleSheet("QGroupBox { border: 1px solid gray; padding: 5px; }")

        # Ajustar o layout do grupo para reduzir o espaço entre o título e a moldura
        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(group_box)
        group_box_layout.setContentsMargins(0, 0, 0, 0)

        # Adiciona o grupo com layout ajustado ao layout fornecido
        layout.addLayout(group_box_layout)

    def create_example_plot(self, layout):
        # Exemplo de dados
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55]
        })

        # Criar a figura e o eixo
        fig, ax = plt.subplots()
        ax.plot(df['x'], df['y'])

        # Configurar o gráfico
        ax.set_title('Exemplo de Gráfico')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        # Adicionar o gráfico ao layout usando FigureCanvas
        canvasGrafico = FigureCanvas(fig)
        layout.addWidget(canvasGrafico)

    def on_button_click(self):
        print('botão clicado')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
