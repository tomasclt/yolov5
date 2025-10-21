# -*- coding: utf-8 -*-
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# -----------------------------
# Configuración de página
# -----------------------------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# Estilos (solo estética)
# -----------------------------
st.markdown("""
<style>
:root{
  --bg:#0b1120; --bg2:#0f172a;
  --panel:#111827; --border:#1f2937;
  --text:#f8fafc; --muted:#cbd5e1;
  --accent:#22d3ee; --accent2:#6366f1;
  --good:#10b981; --warn:#f59e0b; --bad:#ef4444;
}

/* Fondo y tipografía */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 10% 0%, #0f172a 0%, transparent 60%),
              radial-gradient(900px 500px at 90% 0%, #0c1833 0%, transparent 60%),
              linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%) !important;
  color: var(--text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial;
}
main .block-container{ padding-top: 1.4rem; padding-bottom: 2rem; }

/* Títulos */
h1,h2,h3{ color:#f9fafb !important; letter-spacing:-.02em; }
h1 span.grad {
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* Tarjetas */
.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.1rem 1.25rem;
  box-shadow: 0 20px 50px rgba(0,0,0,.45);
  animation: fadeIn .5s ease;
}
@keyframes fadeIn { from {opacity:0; transform: translateY(10px);} to {opacity:1; transform: none;} }

/* Controles */
.stTextInput input, .stNumberInput input, .stSlider, .stCheckbox, .stRadio {
  color: var(--text) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"]{ background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] > div > div{ background: #1f2a44 !important; }
.stTextInput input, .stNumberInput input{
  background: #0f172a !important;
  border: 1px solid #334155 !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  transition: all .22s ease;
}
.stTextInput input:hover, .stNumberInput input:hover{ border-color:#3b82f6 !important; background:#132036 !important; }
.stTextInput input:focus, .stNumberInput input:focus{
  border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(34,211,238,.25);
  background:#0d1829 !important;
}

/* Botones */
.stButton > button{
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  border: 0; color: #fff; font-weight: 600;
  border-radius: 999px; padding: .72rem 1.1rem;
  box-shadow: 0 12px 36px rgba(99,102,241,.35);
  transition: all .18s ease;
}
.stButton > button:hover{ transform: translateY(-1px); box-shadow: 0 16px 46px rgba(99,102,241,.45); }

/* Dataframe */
.dataframe th { background:#1e293b !important; color:#f1f5f9 !important; }
.dataframe td { color:#e2e8f0 !important; }

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child{
  background: #0c1324; border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] *{ color: var(--text) !important; }

/* Badges */
.badge{ display:inline-block; padding:.24rem .55rem; border-radius:999px; font-weight:700; font-size:.8rem;
        border:1px solid rgba(255,255,255,.12); }
.badge-ok{ background: rgba(16,185,129,.15); color:#86efac; border-color: rgba(16,185,129,.35); }
.badge-warn{ background: rgba(245,158,11,.15); color:#fde68a; border-color: rgba(245,158,11,.35); }

/* Imagen con marco */
.frame{
  border:1px solid #1f2937; border-radius:16px; overflow:hidden;
  box-shadow: 0 18px 50px rgba(0,0,0,.5);
}

/* Toast fix (legible) */
[data-testid="stToast"] *{ color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Carga del modelo (misma lógica)
# -----------------------------
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("Intentando método alternativo de carga…")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
1) Verifica PyTorch/YOLOv5 compatibles:
   pip install torch==1.12.0 torchvision==0.13.0
   pip install yolov5==7.0.9
2) Revisa la ruta del archivo de pesos (yolov5s.pt).
3) Si falla, usa carga desde torch hub (pretrained=True).
""")
        return None

# -----------------------------
# UI
# -----------------------------
st.markdown("<h1>🔍 <span class='grad'>Detección de Objetos</span> en Imágenes</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#cbd5e1'>Usa la cámara, ajusta umbrales en la barra lateral y obtiene detecciones con YOLOv5.</div>",
    unsafe_allow_html=True
)

with st.spinner("Cargando modelo YOLOv5…"):
    model = load_yolov5_model()

if model:
    # Sidebar (misma configuración)
    st.sidebar.title("Parámetros")
    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")

    # Contenedor principal
    st.markdown('<div class="card">', unsafe_allow_html=True)
    picture = st.camera_input("📸 Capturar imagen", key="camera")
    st.markdown('</div>', unsafe_allow_html=True)

    if picture:
        # Decodificar imagen
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Detección
        with st.spinner("Detectando objetos…"):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        # Obtener imagen anotada sin cambiar tu lógica
        annotated = None
        try:
            r = results.render()              # algunos devuelven lista, otros modifican internamente
            if hasattr(results, 'imgs') and results.imgs:
                annotated = results.imgs[0]
            elif isinstance(r, list) and len(r) > 0:
                annotated = r[0]
        except Exception:
            annotated = None
        if annotated is None:
            annotated = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        # Parseo de resultados (tu flujo)
        try:
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            col1, col2 = st.columns([1.3, 1], gap="large")

            with col1:
                st.subheader("Imagen con detecciones")
                st.markdown('<div class="frame">', unsafe_allow_html=True)
                # Mostrar en RGB para consistencia visual
                try:
                    st.image(annotated, use_container_width=True)
                except:
                    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.subheader("Objetos detectados")

                label_names = model.names
                category_count = {}
                for category in categories:
                    category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[category_idx] = category_count.get(category_idx, 0) + 1

                data = []
                for category, count in category_count.items():
                    label = label_names[category]
                    confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                    data.append({
                        "Categoría": label,
                        "Cantidad": count,
                        "Confianza promedio": f"{confidence:.2f}"
                    })

                if data:
                    df = pd.DataFrame(data).sort_values("Cantidad", ascending=False)
                    st.dataframe(df, use_container_width=True)
                    # Grafiquito rápido
                    st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    st.toast("Detecciones listas ✅", icon="✅")
                    st.balloons()  # pequeña celebración
                else:
                    st.info("No se detectaron objetos con los parámetros actuales.")
                    st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")

        except Exception as e:
            st.error(f"Error al procesar los resultados: {str(e)}")
            st.stop()
else:
    st.error("No se pudo cargar el modelo. Verifica dependencias e inténtalo nuevamente.")
    st.stop()

# Footer
st.markdown("---")
st.caption("Hecho con Streamlit + YOLOv5 + PyTorch • UI moderna con animaciones ✨")
