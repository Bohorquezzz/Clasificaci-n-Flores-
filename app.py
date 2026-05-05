import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo MLP y el LabelEncoder
try:
    mlp_model = joblib.load('mlp_model.joblib')
    label_encoder = joblib.load('label_encoder_species.joblib')
except FileNotFoundError:
    st.error("Error: Los archivos 'mlp_model.joblib' o 'label_encoder_species.joblib' no se encontraron.")
    st.stop()

# Título de la aplicación
st.title('Predicción de Especies de Iris con Red Neuronal')
st.write('Introduce las características de la flor para predecir su especie.')

# Entradas del usuario
sepal_length = st.number_input('Longitud del sépalo (cm)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input('Ancho del sépalo (cm)', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input('Longitud del pétalo (cm)', min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petal_width = st.number_input('Ancho del pétalo (cm)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Botón para realizar la predicción
if st.button('Predecir Especie'):
    # Crear un DataFrame con las características de entrada
    new_flower_features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                       columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    # Realizar la predicción
    predicted_species_idx = mlp_model.predict(new_flower_features)

    # Decodificar la predicción
    decoded_species_name = label_encoder.inverse_transform(predicted_species_idx)

    st.success(f"La especie predicha es: **{decoded_species_name[0]}**")

# Instrucciones para ejecutar:
# 1. Guarda el código anterior en un archivo llamado `app.py`.
# 2. Abre una terminal y navega al directorio donde guardaste `app.py`.
# 3. Ejecuta `pip install streamlit` (si no lo tienes instalado).
# 4. Ejecuta `streamlit run app.py`.
