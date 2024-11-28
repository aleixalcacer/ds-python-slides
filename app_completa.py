
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

st.title('Explorador de datos')

st.header('Cargar datos')
file = st.file_uploader('Cargar archivo CSV', type='csv')

if file is None:
    st.warning('Por favor, carga un archivo CSV')
else:
    
    df = pd.read_csv(file) 

    st.header('Filtrado de datos')

    if st.checkbox('Eliminar filas con valores faltantes'):
        df = df.dropna()

    x = st.selectbox('Selecciona una columna para el eje x', df.columns)

    y = st.selectbox('Selecciona una columna para el eje y', df.columns)

    hue = st.selectbox('Selecciona una columna para el color', df.columns)

    st.header('Exploración de datos')

    plot = st.selectbox('Selecciona un tipo de gráfico', ['scatterplot', 'swarmplot', 'kdeplot'])

    fig, ax = plt.subplots()

    getattr(sns, plot)(x=x, y=y, data=df, hue=hue, ax=ax)

    st.pyplot(fig)

    st.header('Modelado de datos')

    if df.isnull().values.any():
        st.warning('Por favor, elimina los valores faltantes')
    else:

        X_ = df[[x, y]]
        y_ = df[hue]

        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=42)

        model = MLPClassifier(max_iter=1000)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        st.success('Modelo entrenado con éxito con un score de ' + str(score))

        st.header('Predicción de datos')

        x_pred = st.number_input('Introduce un valor para ' + x)
        y_pred = st.number_input('Introduce un valor para ' + y)

        prediction = model.predict([[x_pred, y_pred]])
        st.write('Predicción:', prediction[0])
