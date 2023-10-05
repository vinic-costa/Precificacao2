import streamlit as st
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

#pageconfig
st.set_page_config(page_title='Tag2u')

# st.write('Hello!')

#Elementos de texto
st.header('TAG2U (version 2)')

## -- Parametros -- #
#Widgets

# Obter valores do usuário
valormercado = st.number_input(label='Digite o valor_mercado do produto', min_value =0, key ='valor_mercado')
altura = st.number_input(label='Digite a altura do produto', min_value=0, key='altura')
largura = st.number_input(label='Digite a largura do produto', min_value=0, key='largura')
profundidade = st.number_input(label='Digite a profundidade do produto', min_value=0, key='profundidade') 

# Botão de atualização
if st.button('Atualizar'):

    # Criar o DataFrame de entrada
    df_input = pd.DataFrame({
        'valormercado':[valormercado],
        'altura': [altura],    
        'largura': [largura],
        'profundidade': [profundidade]
        
        })

    st.dataframe(df_input)


    # Página Lateral
    st.sidebar.header('PREVISÃO DE PREÇOS  DE PRODUTOS')
    st.sidebar.header('Variáveis')

    ## -- Modelo  -- #
    pkl_file_path = 'modelo_precificacao2_outl.joblib'
    
    # Carregar o arquivo .pkl com joblib
    with open(pkl_file_path, 'rb') as model_file:
        modelo = joblib.load(model_file)
    
    # Verificar o tipo do objeto carregado
    st.write(type(modelo))

    # Verificar os tipos de coluna
    column_types = df_input.dtypes

    # Exibir os tipos de coluna na interface do Streamlit
    st.write("Tipos de Coluna:")
    st.write(column_types)

    # Você também pode exibir informações adicionais, como contagem de valores únicos
    unique_value_counts = df_input.nunique()
    st.write("Contagem de Valores Únicos:")
    st.write(unique_value_counts)

 
    def prediction():
        # Verificar os valores de df_input
        st.write("Valores em df_input:")
        st.dataframe(df_input)

        # Realizar a previsão
        prediction = modelo.predict(df_input)[0]
        return prediction

    # Antes de chamar a função prediction(), aplique o StandardScaler nas colunas desejadas
    #scaler = StandardScaler()
    #columns_to_scale = ['altura', 'largura', 'profundidade']
    #df_input[columns_to_scale] = scaler.fit_transform(df_input[columns_to_scale])

    valor_produto = prediction()
    st.write(valor_produto)
