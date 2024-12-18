# Sistema de Detección de Objetos con YOLOv5 y Streamlit

Una aplicación web interactiva para detección de objetos en tiempo real utilizando YOLOv5 y Streamlit. La aplicación permite a los usuarios capturar imágenes a través de su cámara web y realizar detección de objetos con parámetros ajustables.


## 🚀 Características

- Detección de objetos en tiempo real
- Interfaz web intuitiva y responsiva
- Captura de imágenes mediante cámara web
- Parámetros ajustables de detección
- Visualización de resultados en tiempo real
- Estadísticas de detección
- Más de 80 clases de objetos detectables

## 📋 Requisitos previos

- Python 3.9+
- Pip (gestor de paquetes de Python)
- Cámara web

## 🛠️ Instalación

1. Clone este repositorio:
```bash
git clone https://github.com/your-username/object-detection-app.git
cd object-detection-app
```

2. Cree un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows use: venv\Scripts\activate
```

3. Instale las dependencias:
```bash
pip install -r requirements.txt
```

## 📦 Dependencias principales

```txt
streamlit==1.28.0
yolov5==7.0.12
opencv-python==4.8.1
numpy==1.24.3
pandas==2.1.1
```

## 🚀 Uso

1. Active el entorno virtual si lo está usando:
```bash
source venv/bin/activate  # En Windows use: venv\Scripts\activate
```

2. Ejecute la aplicación:
```bash
streamlit run app.py
```

3. Abra su navegador web y vaya a la dirección que muestra Streamlit (generalmente http://localhost:8501)

## 💡 Cómo usar la aplicación

1. **Ajuste los parámetros (opcional)**
   - Umbral de Confianza: Ajusta la sensibilidad de las detecciones
   - Umbral IoU: Controla la superposición permitida entre detecciones

2. **Capture una imagen**
   - Haga clic en el botón de captura
   - Permita el acceso a la cámara web cuando se solicite

3. **Visualice los resultados**
   - Ver las detecciones marcadas en la imagen
   - Consultar el resumen de objetos detectados
   - Revisar las estadísticas de detección

## ⚙️ Configuración

Los principales parámetros ajustables son:

- **Umbral de Confianza** (0.0 - 1.0)
  - Valor predeterminado: 0.25
  - Mayor valor = detecciones más seguras pero posiblemente menos objetos detectados

- **Umbral IoU** (0.0 - 1.0)
  - Valor predeterminado: 0.45
  - Mayor valor = menos superposición entre detecciones

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor, siéntase libre de:

1. Fork el repositorio
2. Crear una rama para su característica (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE.md](LICENSE.md) para más detalles.

## 👏 Agradecimientos

- [YOLOv5](https://github.com/ultralytics/yolov5) por el modelo de detección de objetos
- [Streamlit](https://streamlit.io/) por el framework web
- La comunidad de código abierto por sus invaluables contribuciones

## 📞 Contacto

Carlos Mario Correa - cmcorrea4@gmail.com

Link del Proyecto: [https://github.com/cmcorrea4/object-detection-app](https://github.com/your-username/object-detection-app)
