me das un readme para github con shields y posible nombre de proyecto? import os
import cv2
from ultralytics import YOLO
import shutil
import glob
import numpy as np
import torch
import time

print("Paso 0: Librerías importadas correctamente.")

# ==============================================================================
# PASO 1: CONFIGURACIÓN GLOBAL DEL PROYECTO
# ==============================================================================
print("\nPaso 1: Configurando rutas y parámetros globales...")

# --- Rutas de Archivos ---
MODEL_PATH_VEHICLES = 'F:/Documents/PycharmProjects/YOLOMiDaS/best.pt'
VIDEO_INPUT_PATH = 'F:/Documents/PycharmProjects/YOLOMiDaS/GH012372_no_audio.mp4'

# --- Configuración de Salida ---
OUTPUT_DIR = "./runs_local/final_video_processing_v2_ignore_handlebar"  # Nueva carpeta para esta versión
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_BASENAME = os.path.splitext(os.path.basename(VIDEO_INPUT_PATH))[0]
VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{VIDEO_BASENAME}_final_annotated_ign_hb.mp4")
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, f"{VIDEO_BASENAME}_alerts_ign_hb.log")

print(f"Modelo de vehículos: {os.path.abspath(MODEL_PATH_VEHICLES)}")
print(f"Video de entrada: {os.path.abspath(VIDEO_INPUT_PATH)}")
print(f"Video de salida se guardará en: {os.path.abspath(VIDEO_OUTPUT_PATH)}")
print(f"Log de alertas en: {os.path.abspath(LOG_FILE_PATH)}")

# --- Parámetros de Detección YOLO ---
YOLO_CONFIDENCE_THRESHOLD = 0.35
COCO_CLASSES_TO_SEEK = ['person', 'bicycle', 'dog']

# --- Parámetros de Estimación de Distancia ---
FOCAL_LENGTH_PX = 700  # ¡¡¡CRÍTICO: CALIBRA TU CÁMARA!!!
REAL_OBJECT_SIZES_M = {
    'car': 1.8, 'threewheel': 1.2, 'bus': 2.5, 'truck': 2.6, 'motorbike': 0.8, 'van': 2.0,
    'person': 0.5, 'bicycle': 0.4, 'dog': 0.3
}  # ¡¡¡AJUSTA ESTOS VALORES!!!
MIDAS_DEPTH_INVERSE_SCALE_TO_METERS = 30.0  # ¡¡¡PLACEHOLDER ARBITRARIO!!! AJUSTA O CALIBRA.

# --- Parámetros de Alerta de Riesgo ---
RISK_DISTANCE_HIGH_M = 5.0
RISK_DISTANCE_MEDIUM_M = 10.0
RISK_PROXIMITY_HORIZONTAL_FACTOR = 0.25
RISK_MIN_VERTICAL_POSITION_FACTOR = 0.5
RISK_MAX_VERTICAL_POSITION_FACTOR = 0.95  # Punto de referencia del ciclista
DANGEROUS_OBJECT_CLASSES = ['car', 'truck', 'bus', 'motorbike', 'van', 'dog']

# --- Parámetros para Ignorar Manillar ---
# ¡¡¡AJUSTA ESTOS VALORES BASADO EN TU CONFIGURACIÓN DE CÁMARA!!!
IGNORE_ZONE_Y_START_FACTOR = 0.80  # Desde el 80% del alto hacia abajo
IGNORE_ZONE_X_MIN_FACTOR = 0.10  # Desde el 30% del ancho
IGNORE_ZONE_X_MAX_FACTOR = 0.90  # Hasta el 70% del ancho
# Clases que podrían ser el manillar si son detectadas por COCO y están en la zona
MANILLAR_POSSIBLE_CLASSES_IN_ZONE = ['bicycle', 'motorbike']

# --- Configuración del Modelo de Profundidad ---
DEPTH_MODEL_TYPE = "MiDaS_small"

# ==============================================================================
# PASO 2: FUNCIONES AUXILIARES
# ==============================================================================
print("\nPaso 2: Definiendo funciones auxiliares...")


# (Las funciones load_yolo_model, load_depth_model, get_coco_target_ids,
#  estimate_distance_by_size, estimate_distance_from_depth_map se mantienen
#  exactamente como en la respuesta anterior. Las omito aquí por brevedad
#  pero DEBEN estar en tu script)
def load_yolo_model(path_or_name, description):
    if path_or_name != 'yolov8n.pt' and not os.path.exists(path_or_name):
        print(f"ERROR: Modelo {description} no encontrado en: {path_or_name}")
        return None
    try:
        model = YOLO(path_or_name)
        print(
            f"Modelo {description} '{path_or_name}' cargado. Clases: {model.names if hasattr(model, 'names') else 'N/A'}")
        return model
    except Exception as e:
        print(f"Error cargando modelo {description} desde '{path_or_name}': {e}")
        return None


def load_depth_model(model_type, device):
    print(f"\nCargando modelo de profundidad: {model_type} en dispositivo {device}...")
    try:
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        if "dpt" in model_type.lower() or "beit" in model_type.lower():
            transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
        else:
            transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).midas_transform
        midas.to(device)
        midas.eval()
        print(f"Modelo de profundidad '{model_type}' cargado exitosamente.")
        return midas, transform, True
    except Exception as e:
        print(f"Error al cargar el modelo de profundidad '{model_type}': {e}")
        print("Se continuará sin estimación de profundidad avanzada.")
        return None, None, False


def get_coco_target_ids(coco_model, target_names):
    ids = []
    if hasattr(coco_model, 'names') and isinstance(coco_model.names, dict):
        for name_to_seek in target_names:
            found_id = None
            for class_id, class_name in coco_model.names.items():
                if class_name == name_to_seek: found_id = class_id; break
            if found_id is not None:
                ids.append(found_id)
            else:
                print(f"ADVERTENCIA: Clase COCO '{name_to_seek}' no encontrada.")
    return ids


def estimate_distance_by_size(obj_width_px, obj_real_width_m, focal_length_px):
    if obj_width_px > 0 and obj_real_width_m > 0 and focal_length_px > 0:
        return (obj_real_width_m * focal_length_px) / obj_width_px
    return -1


def estimate_distance_from_depth_map(depth_map, x1, y1, x2, y2):
    global MIDAS_DEPTH_INVERSE_SCALE_TO_METERS  # Para usar la constante global
    if depth_map is None: return -1, "N/A"
    roi_y1, roi_y2 = np.clip(y1, 0, depth_map.shape[0] - 1), np.clip(y2, 0, depth_map.shape[0] - 1)
    roi_x1, roi_x2 = np.clip(x1, 0, depth_map.shape[1] - 1), np.clip(x2, 0, depth_map.shape[1] - 1)
    if roi_y1 < roi_y2 and roi_x1 < roi_x2:
        roi_depth_patch = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi_depth_patch.size > 0:
            median_depth_val_map = np.median(roi_depth_patch)
            if median_depth_val_map > 1e-5:
                dist_m = MIDAS_DEPTH_INVERSE_SCALE_TO_METERS / median_depth_val_map
                dist_m = max(0.1, min(dist_m, 200.0))
                return dist_m, "M"
    return -1, "N/A"


# ==============================================================================
# PASO 3: CARGAR TODOS LOS MODELOS
# ==============================================================================
print("\nPaso 3: Cargando todos los modelos...")
model_vehicles = load_yolo_model(MODEL_PATH_VEHICLES, "Vehículos")
model_coco = load_yolo_model('yolov8n.pt', "COCO")
coco_target_ids = get_coco_target_ids(model_coco, COCO_CLASSES_TO_SEEK) if model_coco else []
device_depth = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas_model, midas_transform, depth_model_ready = load_depth_model(DEPTH_MODEL_TYPE, device_depth)

if not model_vehicles:
    print("ERROR CRÍTICO: No se pudo cargar el modelo de vehículos. Abortando.");
    exit()

# ==============================================================================
# PASO 4: INICIAR PROCESAMIENTO DE VIDEO
# ==============================================================================
print("\nPaso 4: Iniciando procesamiento de video...")
if not os.path.exists(VIDEO_INPUT_PATH): raise FileNotFoundError(f"Video no encontrado: {VIDEO_INPUT_PATH}")
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened(): raise IOError(f"No se pudo abrir video: {VIDEO_INPUT_PATH}")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_out = fps_in if fps_in > 0 else 30.0
print(f"Video: {frame_w}x{frame_h} @ {fps_in:.2f} FPS, Frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps_out, (frame_w, frame_h))
if not out_video.isOpened(): cap.release(); raise IOError(f"No se pudo abrir VideoWriter para: {VIDEO_OUTPUT_PATH}")

print(f"Procesando y guardando en: {VIDEO_OUTPUT_PATH}")
frame_num = 0;
frames_written = 0;
log_alerts = []
cyclist_ref_point = (frame_w // 2, int(frame_h * RISK_MAX_VERTICAL_POSITION_FACTOR))

# Calcular límites de la zona de ignorar una vez
ignore_zone_y_px_start = int(frame_h * IGNORE_ZONE_Y_START_FACTOR)
ignore_zone_x_px_min = int(frame_w * IGNORE_ZONE_X_MIN_FACTOR)
ignore_zone_x_px_max = int(frame_w * IGNORE_ZONE_X_MAX_FACTOR)

start_time_proc = time.time()
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: print(f"Fin del video o error en frame {frame_num + 1}."); break
        frame_num += 1
        annotated_frame = frame.copy()
        current_frame_detections = []

        # --- Inferencia YOLO ---
        preds_v = model_vehicles.predict(source=frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        if preds_v and preds_v[0].boxes:
            for box in preds_v[0].boxes: current_frame_detections.append(
                {"xywh": box.xywh.cpu().numpy()[0], "cls_id": int(box.cls.cpu().item()),
                 "conf": float(box.conf.cpu().item()), "model_names_map": model_vehicles.names,
                 "source_model": "vehicles"})
        if model_coco and coco_target_ids:
            preds_c = model_coco.predict(source=frame, conf=YOLO_CONFIDENCE_THRESHOLD, classes=coco_target_ids,
                                         verbose=False)
            if preds_c and preds_c[0].boxes:
                for box in preds_c[0].boxes: current_frame_detections.append(
                    {"xywh": box.xywh.cpu().numpy()[0], "cls_id": int(box.cls.cpu().item()),
                     "conf": float(box.conf.cpu().item()), "model_names_map": model_coco.names, "source_model": "coco"})

        # --- Estimación de Profundidad MiDaS ---
        depth_map = None
        if depth_model_ready:
            try:
                img_rgb_d = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_b_d = midas_transform(img_rgb_d).to(device_depth)
                with torch.no_grad():
                    pred_d = midas_model(input_b_d)
                    pred_d = torch.nn.functional.interpolate(pred_d.unsqueeze(1), size=img_rgb_d.shape[:2],
                                                             mode="bicubic", align_corners=False).squeeze()
                depth_map = pred_d.cpu().numpy()
            except Exception:
                depth_map = None  # Silenciar error por frame

        cv2.circle(annotated_frame, cyclist_ref_point, 8, (255, 0, 255), -1)

        for det in current_frame_detections:
            x_c, y_c, w_obj, h_obj = det["xywh"];
            cls_id = det["cls_id"];
            conf = det["conf"]
            model_names = det["model_names_map"];
            label_name = model_names[cls_id]
            x1, y1, x2, y2 = int(x_c - w_obj / 2), int(y_c - h_obj / 2), int(x_c + w_obj / 2), int(y_c + h_obj / 2)

            # --- Lógica para Ignorar Manillar ---
            obj_center_y_for_ignore = y_c  # Podrías usar y2 si es mejor
            obj_center_x_for_ignore = x_c
            is_likely_handlebar = False
            if obj_center_y_for_ignore > ignore_zone_y_px_start and \
                    obj_center_x_for_ignore > ignore_zone_x_px_min and \
                    obj_center_x_for_ignore < ignore_zone_x_px_max:
                if label_name in MANILLAR_POSSIBLE_CLASSES_IN_ZONE:  # Solo si es una clase que podría ser el manillar
                    is_likely_handlebar = True

            if is_likely_handlebar:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.putText(annotated_frame, f"{label_name}(Ign)", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA)
                continue  # Saltar el resto del procesamiento para esta detección
            # --- Fin Lógica Ignorar Manillar ---

            dist_m, dist_src = estimate_distance_from_depth_map(depth_map, x1, y1, x2, y2)
            if dist_m <= 0:
                obj_real_w = REAL_OBJECT_SIZES_M.get(label_name, -1)
                dist_m_size = estimate_distance_by_size(w_obj, obj_real_w, FOCAL_LENGTH_PX)
                if dist_m_size > 0: dist_m = dist_m_size; dist_src = "S"

            is_high_risk = False;
            is_medium_risk = False;
            alert_msg = ""
            box_color = (0, 180, 0) if det["source_model"] == "vehicles" else (
            180, 100, 0)  # Colores un poco más brillantes

            if label_name in DANGEROUS_OBJECT_CLASSES and dist_m > 0:
                horiz_dist_px = abs(x_c - cyclist_ref_point[0])
                obj_bottom_y = y2
                in_cone_h = horiz_dist_px < (frame_w * RISK_PROXIMITY_HORIZONTAL_FACTOR)
                in_zone_v = (obj_bottom_y > frame_h * RISK_MIN_VERTICAL_POSITION_FACTOR) and \
                            (obj_bottom_y < cyclist_ref_point[1] + frame_h * 0.05)
                if in_cone_h and in_zone_v:
                    if dist_m < RISK_DISTANCE_HIGH_M:
                        is_high_risk = True;
                        alert_msg = f"ALERTA! {label_name} {dist_m:.1f}m";
                        box_color = (0, 0, 255)
                        log_alerts.append(f"F{frame_num}: {alert_msg}");
                        print(f"CONSOLE F{frame_num}: {alert_msg}")
                    elif dist_m < RISK_DISTANCE_MEDIUM_M:
                        is_medium_risk = True;
                        alert_msg = f"Precaucion {label_name} {dist_m:.1f}m";
                        box_color = (0, 165, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2 if (is_high_risk or is_medium_risk) else 1)
            label_text = f"{label_name} {conf:.1f}"  # Confianza con 1 decimal
            if dist_m > 0: label_text += f" {dist_m:.1f}m({dist_src})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)  # Fuente aún más pequeña
            cv2.rectangle(annotated_frame, (x1, y1 - th - 2), (x1 + tw, y1 - 1), box_color, -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)
            if alert_msg and (is_high_risk or is_medium_risk):
                cv2.putText(annotated_frame, alert_msg.split('! ')[-1] if '!' in alert_msg else alert_msg,
                            (x1, y2 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2 if is_high_risk else 1,
                            cv2.LINE_AA)

        out_video.write(annotated_frame)
        frames_written += 1
        if frame_num % (int(fps_out) * 10) == 0:
            elapsed = time.time() - start_time_proc;
            fps_proc = frame_num / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_num}/{total_frames}... Escritura FPS: {fps_proc:.2f}")
except KeyboardInterrupt:
    print("\nProcesamiento interrumpido.")
except Exception as e:
    print(f"Error en bucle: {e}"); import traceback; traceback.print_exc()
finally:
    total_time = time.time() - start_time_proc
    avg_fps_total = frame_num / total_time if total_time > 0 else 0
    print(
        f"\nCerrando. Tiempo total: {total_time:.2f}s. FPS prom: {avg_fps_total:.2f}. Frames leídos: {frame_num}, Escritos: {frames_written}.")
    if cap.isOpened(): cap.release()
    if out_video.isOpened(): out_video.release()
    cv2.destroyAllWindows()

if os.path.exists(VIDEO_OUTPUT_PATH) and frames_written > 0:
    print(f"\n✅ Video procesado: {os.path.abspath(VIDEO_OUTPUT_PATH)}")
else:
    print("\n⚠️ No se guardó video de salida o no se escribieron frames.")
if log_alerts:
    with open(LOG_FILE_PATH, 'w') as f_log: f_log.write("\n".join(log_alerts))
    print(f"Log de alertas: {os.path.abspath(LOG_FILE_PATH)}")
print("\n--- Proceso Finalizado ---")
