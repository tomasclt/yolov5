import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Sistema de Detección de Objetos",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .css-1v0mbdj.etr89bj1 {
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/yolov5-logo.png", width=200)
    st.title("Configuración")
    
    # Model parameters
    st.subheader("Parámetros del Modelo")
    model_conf = st.slider(
        'Umbral de Confianza',
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        help="Ajusta el nivel de confianza mínimo para las detecciones"
    )
    
    model_iou = st.slider(
        'Umbral IoU',
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        help="Ajusta el umbral de Intersección sobre Unión para NMS"
    )
    
    st.markdown("---")
    
    # Information section
    st.subheader("Información")
    st.info("""
    Este sistema utiliza YOLOv5 para la detección de objetos en tiempo real.
    
    **Características principales:**
    - Detección en tiempo real
    - Más de 80 clases de objetos
    - Ajuste de parámetros flexible
    """)
    
    st.markdown("---")
    
    # Help section
    with st.expander("❓ Ayuda"):
        st.markdown("""
        **Cómo usar:**
        1. Ajusta los parámetros según necesites
        2. Captura una foto con la cámara
        3. Espera los resultados del análisis
        
        **Parámetros:**
        - **Confianza**: Mayor valor = detecciones más seguras
        - **IoU**: Mayor valor = menos superposición entre detecciones
        """)

# Main content
st.title("🔍 Sistema de Detección de Objetos")

# Initialize model
@st.cache_resource
def load_model():
    model = yolov5.load('yolov5s.pt')
    return model

model = load_model()
model.conf = model_conf
model.iou = model_iou
model.agnostic = False
model.multi_label = False
model.max_det = 1000

# Camera input with custom styling
st.markdown("### 📸 Captura de Imagen")
picture = st.camera_input("", label_visibility='collapsed')

if picture:
    with st.spinner('Procesando imagen...'):
        # Process image
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference
        results = model(cv2_img)
        predictions = results.pred[0]
        boxes = predictions[:, :4] 
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Detecciones")
            results.render()
            st.image(cv2_img, channels='BGR', use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Resumen")
            
            # Crear un diccionario para contar las categorías
            category_counts = {}
            
            # Procesar las detecciones y contar por categoría
            for category in categories:
                category_name = model.names[int(category)]
                if category_name in category_counts:
                    category_counts[category_name] += 1
                else:
                    category_counts[category_name] = 1
            
            # Convertir el diccionario a DataFrame
            if category_counts:
                df_sum = pd.DataFrame([
                    {"Categoría": cat, "Cantidad": count} 
                    for cat, count in category_counts.items()
                ]).sort_values("Cantidad", ascending=False)
                
                # Show results with better styling
                st.dataframe(
                    df_sum,
                    hide_index=True,
                    use_container_width=True
                )
                
                # Add some statistics
                st.markdown("#### 📈 Estadísticas")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Objetos", df_sum["Cantidad"].sum())
                with col2:
                    st.metric("Tipos Diferentes", len(df_sum))
            else:
                st.warning("No se detectaron objetos en la imagen. Intenta ajustar el umbral de confianza o tomar una nueva foto.")
                st.metric("Total Objetos", 0)
