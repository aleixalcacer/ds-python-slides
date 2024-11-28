
import streamlit as st

st.title('Mi primera aplicación con Streamlit')

numero = st.slider('Selecciona un número', 0, 100)
st.write(f'Has seleccionado el número {numero}')

if st.button('Haz clic aquí'):
    st.write('¡Has hecho clic!')
