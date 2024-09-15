import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import mysql.connector

# Credenciais do banco de dados MySQL usando o endereço público
host = 'autorack.proxy.rlwy.net'
usuario = 'root'
senha = 'VqMXhctmmBdpdlusjytjGTxGiaPtWXEn'
banco_de_dados = 'railway'
porta = 29968

# Função para conectar ao banco de dados MySQL
def conectar_banco_de_dados():
    """Conecta ao banco de dados MySQL e retorna a conexão."""
    try:
        connection = mysql.connector.connect(
            host=host,
            user=usuario,
            password=senha,
            database=banco_de_dados,
            port=porta
        )
        return connection
    except mysql.connector.Error as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

# Função para criar a tabela se não existir
def criar_tabela():
    connection = conectar_banco_de_dados()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imoveis (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tamanho FLOAT,
                condicao INT,
                quartos INT,
                banheiros INT,
                bairro VARCHAR(255),
                tipo_imovel VARCHAR(255),
                preco FLOAT
            )
        """)
        connection.commit()
        cursor.close()
        connection.close()
        st.success("Tabela criada com sucesso!")
    else:
        st.error("Não foi possível conectar ao banco de dados para criar a tabela.")

# Função para inserir dados no banco de dados
def inserir_dado(data):
    connection = conectar_banco_de_dados()
    if connection is not None:
        try:
            cursor = connection.cursor()
            sql = """
                INSERT INTO imoveis (tamanho, condicao, quartos, banheiros, bairro, tipo_imovel, preco)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                data['tamanho'],
                data['condicao'],
                data['quartos'],
                data['banheiros'],
                data['bairro'],
                data['tipo_imovel'],
                data['preco']
            )
            cursor.execute(sql, values)
            connection.commit()
            cursor.close()
            connection.close()
            st.success("Dados inseridos com sucesso!")
        except mysql.connector.Error as e:
            st.error(f"Erro ao inserir dados no banco de dados: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")
        finally:
            if connection.is_connected():
                connection.close()
    else:
        st.error("Não foi possível conectar ao banco de dados para inserir dados.")
    st.rerun()

# Função para recuperar dados do banco de dados
def recuperar_dados():
    connection = conectar_banco_de_dados()
    if connection is not None:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM imoveis")
            result = cursor.fetchall()
            cursor.close()
            connection.close()
            if result:
                data = pd.DataFrame(result)
                return data
            else:
                return pd.DataFrame()
        except mysql.connector.Error as e:
            st.error(f"Erro ao recuperar dados do banco de dados: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")
    else:
        st.error("Não foi possível conectar ao banco de dados para recuperar dados.")
        return pd.DataFrame()

# Função para criar dataset sintético
def create_synthetic_dataset():
    np.random.seed(42)
    data = pd.DataFrame({
        'tamanho': np.random.randint(50, 500, size=1000),
        'condicao': np.random.randint(1, 11, size=1000),
        'quartos': np.random.randint(1, 6, size=1000),
        'banheiros': np.random.randint(1, 4, size=1000),
        'bairro': np.random.choice(['Centro', 'Zona Sul', 'Zona Norte', 'Zona Leste', 'Zona Oeste'], size=1000),
        'tipo_imovel': np.random.choice(['Casa', 'Apartamento'], size=1000)
    })
    data['preco'] = (
        data['tamanho'] * 50 +
        data['condicao'] * 100 +
        data['quartos'] * 200 +
        data['banheiros'] * 150 +
        data['bairro'].map({
            'Centro': 1000,
            'Zona Sul': 800,
            'Zona Norte': 600,
            'Zona Leste': 400,
            'Zona Oeste': 200
        }) +
        data['tipo_imovel'].map({
            'Casa': 500,
            'Apartamento': 0
        }) +
        np.abs(np.random.normal(0, 50000, size=1000))
    )
    return data

# Função para inserir múltiplos dados no banco de dados
def inserir_dados_em_lote(dados):
    connection = conectar_banco_de_dados()
    if connection is not None:
        try:
            cursor = connection.cursor()
            sql = """
                INSERT INTO imoveis (tamanho, condicao, quartos, banheiros, bairro, tipo_imovel, preco)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            for _, row in dados.iterrows():
                values = (
                    row['tamanho'],
                    row['condicao'],
                    row['quartos'],
                    row['banheiros'],
                    row['bairro'],
                    row['tipo_imovel'],
                    row['preco']
                )
                cursor.execute(sql, values)
            connection.commit()
            cursor.close()
            connection.close()
            st.success("Dados aleatórios inseridos na tabela!")
        except mysql.connector.Error as e:
            st.error(f"Erro ao inserir dados no banco de dados: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")
        finally:
            if connection.is_connected():
                connection.close()
    else:
        st.error("Não foi possível conectar ao banco de dados para inserir dados.")


# Função para verificar se a tabela está vazia e inserir dados sintéticos se necessário
def verificar_e_inserir_dados():
    dados = recuperar_dados()
    if dados.empty:
        st.info("A tabela está vazia. Inserindo dados aleatórios...")
        dados_aleatorios = create_synthetic_dataset()
        inserir_dados_em_lote(dados_aleatorios)
    else:
        st.info("A tabela já contém dados.")

# Função para re-treinar o modelo
def retrain_model():
    verificar_e_inserir_dados()
    data = recuperar_dados()
    if data.empty:
        data = create_synthetic_dataset()
    else:
        data['condicao'] = data['condicao'].astype(int)
        data['quartos'] = data['quartos'].astype(int)
        data['banheiros'] = data['banheiros'].astype(int)
        data['tamanho'] = data['tamanho'].astype(float)
        data['preco'] = data['preco'].astype(float)
        
    if 'id' in data.columns:
        X = data.drop('id', axis=1)
    else:
        X = data.copy()
    y = data['preco']
    
    numerical_features = ['tamanho', 'condicao', 'quartos', 'banheiros']
    categorical_features = ['bairro', 'tipo_imovel']
    numerical_transformer = 'passthrough'
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
    global final_model
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    final_model.fit(X, y)
    st.rerun()

# Inicializar o modelo
if 'final_model' not in st.session_state:
    criar_tabela()
    retrain_model()
    st.session_state['final_model'] = final_model
    st.rerun()
else:
    final_model = st.session_state['final_model']

# Sidebar para navegação
pagina = st.sidebar.selectbox("Navegação", ["Previsor de Preços", "Editar Tabela"])

if pagina == "Previsor de Preços":
    st.title("Previsor de Preços de Imóveis")
    st.write("Insira as características do imóvel para obter uma estimativa de preço.")

    tamanho = st.number_input("Tamanho do imóvel (m²)", min_value=10, max_value=1000, value=100)
    condicao = st.slider("Condição do imóvel (0 - Péssimo, 10 - Excelente)", 0, 10, 5)
    quartos = st.number_input("Número de quartos", min_value=1, max_value=10, value=3)
    banheiros = st.number_input("Número de banheiros", min_value=1, max_value=5, value=2)
    bairro = st.selectbox("Bairro", ['Centro', 'Zona Sul', 'Zona Norte', 'Zona Leste', 'Zona Oeste'])
    tipo_imovel = st.selectbox("Tipo do Imóvel", ['Casa', 'Apartamento'])

    if st.button("Prever Preço"):
        input_data = pd.DataFrame({
            'tamanho': [tamanho],
            'condicao': [condicao],
            'quartos': [quartos],
            'banheiros': [banheiros],
            'bairro': [bairro],
            'tipo_imovel': [tipo_imovel]
        })
        predicted_price = final_model.predict(input_data)[0]
        st.success(f"O preço estimado é R$ {predicted_price:,.2f}")
        
        real_price = st.number_input("Se o preço estiver incorreto, insira o valor real:")
        if st.button("Enviar Preço Real"):
            if real_price > 0:
                new_data = {
                    'tamanho': tamanho,
                    'condicao': condicao,
                    'quartos': quartos,
                    'banheiros': banheiros,
                    'bairro': bairro,
                    'tipo_imovel': tipo_imovel,
                    'preco': real_price
                }
                inserir_dado(new_data)
                st.success("Obrigado! Seus dados ajudarão a melhorar nossas previsões.")
                retrain_model()
                st.session_state['final_model'] = final_model
            else:
                st.warning("Por favor, insira um preço real válido.")

elif pagina == "Editar Tabela":
    st.title("Editar Tabela de Imóveis")
    
    dados = recuperar_dados()
    if not dados.empty:
        st.dataframe(dados)
        
        id_editar = st.number_input("ID do registro para editar", min_value=1, value=1)
        registro = dados[dados['id'] == id_editar]
        
        if not registro.empty:
            st.write("Edite os valores abaixo e clique em 'Atualizar'.")
            tamanho_edit = st.number_input("Tamanho do imóvel (m²)", min_value=10, max_value=1000, value=int(registro['tamanho'].values[0]))
            condicao_edit = st.slider("Condição do imóvel (0 - Péssimo, 10 - Excelente)", 0, 10, int(registro['condicao'].values[0]))
            quartos_edit = st.number_input("Número de quartos", min_value=1, max_value=10, value=int(registro['quartos'].values[0]))
            banheiros_edit = st.number_input("Número de banheiros", min_value=1, max_value=5, value=int(registro['banheiros'].values[0]))
            bairro_edit = st.selectbox("Bairro", ['Centro', 'Zona Sul', 'Zona Norte', 'Zona Leste', 'Zona Oeste'], index=['Centro', 'Zona Sul', 'Zona Norte', 'Zona Leste', 'Zona Oeste'].index(registro['bairro'].values[0]))
            tipo_imovel_edit = st.selectbox("Tipo do Imóvel", ['Casa', 'Apartamento'], index=['Casa', 'Apartamento'].index(registro['tipo_imovel'].values[0]))
            preco_edit = st.number_input("Preço do imóvel", min_value=0.0, value=float(registro['preco'].values[0]))

            if st.button("Atualizar"):
                connection = conectar_banco_de_dados()
                if connection is not None:
                    try:
                        cursor = connection.cursor()
                        cursor.execute("""
                            UPDATE imoveis
                            SET tamanho = %s, condicao = %s, quartos = %s, banheiros = %s, bairro = %s, tipo_imovel = %s, preco = %s
                            WHERE id = %s
                        """, (tamanho_edit, condicao_edit, quartos_edit, banheiros_edit, bairro_edit, tipo_imovel_edit, preco_edit, id_editar))
                        connection.commit()
                        cursor.close()
                        connection.close()
                        st.success("Registro atualizado com sucesso!")
                        st.rerun()
                    except mysql.connector.Error as e:
                        st.error(f"Erro ao atualizar o registro: {e}")
                    except Exception as e:
                        st.error(f"Ocorreu um erro inesperado: {e}")
                    finally:
                        if connection.is_connected():
                            connection.close()
                else:
                    st.error("Não foi possível conectar ao banco de dados para atualizar o registro.")
        else:
            st.warning("ID não encontrado.")
    else:
        st.warning("A tabela está vazia. Nenhum dado disponível para edição.")
    
    # Formulário de Inclusão de Registro dentro de um Expander
    with st.expander("Incluir Novo Registro"):
        st.write("Preencha os dados do novo imóvel para adicionar ao banco de dados.")
        tamanho_novo = st.number_input("Tamanho do imóvel (m²)", min_value=10, max_value=1000, value=100, key='tamanho_novo')
        condicao_novo = st.slider("Condição do imóvel (0 - Péssimo, 10 - Excelente)", 0, 10, 5, key='condicao_novo')
        quartos_novo = st.number_input("Número de quartos", min_value=1, max_value=10, value=3, key='quartos_novo')
        banheiros_novo = st.number_input("Número de banheiros", min_value=1, max_value=5, value=2, key='banheiros_novo')
        bairro_novo = st.selectbox("Bairro", ['Centro', 'Zona Sul', 'Zona Norte', 'Zona Leste', 'Zona Oeste'], key='bairro_novo')
        tipo_imovel_novo = st.selectbox("Tipo do Imóvel", ['Casa', 'Apartamento'], key='tipo_imovel_novo')
        preco_novo = st.number_input("Preço do imóvel", min_value=0.0, value=100000.0, key='preco_novo')

        if st.button("Adicionar Novo Registro"):
            novo_registro = {
                'tamanho': tamanho_novo,
                'condicao': condicao_novo,
                'quartos': quartos_novo,
                'banheiros': banheiros_novo,
                'bairro': bairro_novo,
                'tipo_imovel': tipo_imovel_novo,
                'preco': preco_novo
            }
            inserir_dado(novo_registro)
            retrain_model()
            st.session_state['final_model'] = final_model
            st.success("Novo registro adicionado e modelo re-treinado com sucesso!")
            st.rerun()
