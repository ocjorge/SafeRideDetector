# 🚴‍♂️ CycleGuard - AI-Powered Cycling Safety System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)

Un sistema inteligente de análisis de video que detecta vehículos, peatones y otros objetos peligrosos en grabaciones de ciclismo, proporcionando alertas de seguridad en tiempo real con estimación de distancias.

## ✨ Características Principales

- 🎯 **Detección Multi-Modelo**: Combina modelos YOLO personalizados para vehículos y COCO para objetos generales
- 📏 **Estimación de Distancia Dual**: 
  - Método basado en tamaño de objetos con calibración de cámara
  - Estimación de profundidad usando MiDaS (Monocular Depth Estimation)
- ⚠️ **Sistema de Alertas Inteligente**: Clasificación de riesgo (Alto/Medio) basado en proximidad y zona de peligro
- 🚲 **Filtrado de Manillar**: Ignora automáticamente detecciones del propio manillar de la bicicleta
- 📊 **Análisis Visual**: Video anotado con cajas delimitadoras, distancias y alertas
- 📝 **Registro de Eventos**: Log detallado de todas las alertas de seguridad

## 🛠️ Tecnologías Utilizadas

- **YOLO v8** (Ultralytics) - Detección de objetos en tiempo real
- **MiDaS** (Intel) - Estimación de profundidad monocular
- **OpenCV** - Procesamiento de video e imágenes
- **PyTorch** - Framework de deep learning
- **NumPy** - Computación numérica

## 📋 Requisitos del Sistema

### Hardware Recomendado
- GPU NVIDIA con soporte CUDA (opcional pero recomendado)
- 8GB+ RAM
- Espacio en disco suficiente para videos de entrada y salida

### Software
```
Python 3.8+
CUDA Toolkit (opcional, para aceleración GPU)
```

## 🚀 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/cycleguard.git
cd cycleguard
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Dependencias Principales
```txt
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
```

## ⚙️ Configuración

### 1. Modelos Requeridos

**Modelo de Vehículos Personalizado:**
- Entrena tu propio modelo YOLO o descarga uno pre-entrenado
- Coloca el archivo `.pt` en la ruta especificada en `MODEL_PATH_VEHICLES`

**Modelo COCO (Automático):**
- Se descarga automáticamente al ejecutar (`yolov8n.pt`)

### 2. Calibración de Cámara

**⚠️ CRÍTICO**: Ajusta estos parámetros según tu configuración:

```python
FOCAL_LENGTH_PX = 700  # Longitud focal en píxeles
REAL_OBJECT_SIZES_M = {
    'car': 1.8,      # Ancho promedio de auto en metros
    'person': 0.5,   # Ancho promedio de persona
    'bicycle': 0.4,  # Ancho promedio de bicicleta
    # ... más objetos
}
```

### 3. Configuración de Rutas

Modifica las rutas en el script principal:

```python
MODEL_PATH_VEHICLES = 'ruta/a/tu/modelo.pt'
VIDEO_INPUT_PATH = 'ruta/a/tu/video.mp4'
OUTPUT_DIR = './resultados'
```

## 🎮 Uso

### Uso Básico
```bash
python cycleguard.py
```

### Parámetros Configurables

**Detección:**
```python
YOLO_CONFIDENCE_THRESHOLD = 0.35  # Umbral de confianza
COCO_CLASSES_TO_SEEK = ['person', 'bicycle', 'dog']  # Clases a detectar
```

**Alertas de Riesgo:**
```python
RISK_DISTANCE_HIGH_M = 5.0    # Distancia de alto riesgo (metros)
RISK_DISTANCE_MEDIUM_M = 10.0  # Distancia de riesgo medio (metros)
```

**Zona de Ignorar Manillar:**
```python
IGNORE_ZONE_Y_START_FACTOR = 0.80  # Desde 80% de altura hacia abajo
IGNORE_ZONE_X_MIN_FACTOR = 0.10    # Desde 10% de ancho
IGNORE_ZONE_X_MAX_FACTOR = 0.90    # Hasta 90% de ancho
```

## 📊 Resultados

El sistema genera:

1. **Video Anotado**: `{video_name}_final_annotated_ign_hb.mp4`
   - Cajas delimitadoras coloreadas por tipo de riesgo
   - Etiquetas con confianza y distancia estimada
   - Punto de referencia del ciclista

2. **Log de Alertas**: `{video_name}_alerts_ign_hb.log`
   - Timestamp de cada alerta
   - Tipo de objeto y distancia
   - Nivel de riesgo

### Código de Colores

| Color | Significado |
|-------|-------------|
| 🟢 Verde | Objetos detectados sin riesgo |
| 🔵 Azul | Objetos COCO (personas, bicicletas, etc.) |
| 🟠 Naranja | Riesgo medio (5-10m en zona de peligro) |
| 🔴 Rojo | **ALTO RIESGO** (<5m en zona de peligro) |
| ⚫ Gris | Objetos ignorados (manillar) |

## 🔧 Personalización

### Agregar Nuevas Clases de Objetos
```python
# En DANGEROUS_OBJECT_CLASSES
DANGEROUS_OBJECT_CLASSES = ['car', 'truck', 'bus', 'motorbike', 'van', 'dog', 'nueva_clase']

# En REAL_OBJECT_SIZES_M
REAL_OBJECT_SIZES_M['nueva_clase'] = 1.5  # tamaño en metros
```

### Ajustar Sensibilidad
```python
# Zona de riesgo más amplia
RISK_PROXIMITY_HORIZONTAL_FACTOR = 0.35  # era 0.25

# Alertas más tempranas
RISK_DISTANCE_HIGH_M = 7.0  # era 5.0
```

## 📈 Rendimiento

**Velocidades típicas (GPU RTX 3070):**
- Video 1080p: ~15-20 FPS
- Video 720p: ~25-30 FPS
- Modo CPU: ~3-5 FPS

**Optimizaciones disponibles:**
- Usar modelo MiDaS más pequeño: `MiDaS_small` vs `DPT_Large`
- Reducir resolución de entrada
- Procesar cada N frames en lugar de todos

## 🤝 Contribuir

1. Fork del proyecto
2. Crear rama de feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📝 TODO

- [ ] Interface gráfica de usuario
- [ ] Procesamiento en tiempo real desde cámara
- [ ] Calibración automática de cámara
- [ ] Soporte para múltiples formatos de video
- [ ] API REST para integración
- [ ] Métricas de rendimiento detalladas
- [ ] Configuración por archivo JSON/YAML

## ⚠️ Limitaciones

- Requiere calibración manual de cámara para distancias precisas
- La estimación de profundidad MiDaS es aproximada
- Rendimiento dependiente de GPU para velocidad óptima
- Funciona mejor con iluminación adecuada

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- [Ultralytics](https://ultralytics.com) por YOLO v8
- [Intel ISL](https://github.com/intel-isl/MiDaS) por MiDaS
- Comunidad de OpenCV
- Contribuidores del proyecto

## 📞 Contacto

- GitHub: [@ocjore](https://github.com/ocjorge)

---

⭐ **¡Si este proyecto te ayuda, considera darle una estrella!** ⭐
