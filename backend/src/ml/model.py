import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import yaml
import os
from datetime import datetime

from .config import ml_config

class YOLOModel:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.load_model()
        # Создаем директорию для сохранения результатов
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self):
        """Загружает модель и классы из YAML"""
        try:
            self.model = YOLO("weights/best2.pt")
            
            with open(ml_config.data_yaml_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                self.class_names = data.get('names', [])
                
            print(f"Модель загружена. Классы: {self.class_names}")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise
    
    def predict(self, image_data: bytes, confidence_threshold: float = 0.25, iou_threshold: float = 0.45) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Выполняет предсказание на изображении"""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Не удалось декодировать изображение")
        
        results = self.model.predict(
            image,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.item())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(box.conf.item()),
                        'bbox': box.xyxy[0].tolist()
                    }
                    detections.append(detection)
        
        return detections, image
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Рисует bounding boxes на изображении"""
        # Цвета для разных классов
        colors = [
            (0, 255, 0),    # зеленый
            (255, 0, 0),    # синий  
            (0, 0, 255),    # красный
            (255, 255, 0),  # голубой
            (255, 0, 255),  # розовый
            (0, 255, 255),  # желтый
            (128, 0, 128),  # фиолетовый
            (255, 165, 0),  # оранжевый
            (0, 128, 128),  # темно-голубой
            (128, 128, 0),  # оливковый
            (128, 0, 0)     # темно-красный
        ]
        
        image_with_boxes = image.copy()
        
        for detection in detections:
            class_id = detection['class_id']
            confidence = detection['confidence']
            bbox = detection['bbox']
            class_name = detection['class_name']
            
            # Выбираем цвет для класса
            color = colors[class_id % len(colors)]
            
            # Координаты bounding box
            x1, y1, x2, y2 = map(int, bbox)
            
            # Рисуем прямоугольник
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Текст с названием класса и уверенностью
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Фон для текста
            cv2.rectangle(image_with_boxes, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Текст
            cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image_with_boxes
    
    def save_annotated_image(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                           filename: str, output_dir: str = None) -> str:
        """Сохраняет изображение с bounding boxes"""
        if output_dir is None:
            # Для одиночных изображений создаем новую директорию
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"session_{timestamp}")
        
        # Убеждаемся, что директория существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Рисуем bounding boxes
        annotated_image = self.draw_detections(image, detections)
        
        # Очищаем имя файла от пути (если есть)
        clean_filename = os.path.basename(filename)
        
        # Сохраняем изображение
        output_path = os.path.join(output_dir, f"annotated_{clean_filename}")
        success = cv2.imwrite(output_path, annotated_image)
        
        if success:
            print(f"Изображение сохранено: {output_path}")
            return output_path
        else:
            print(f"Ошибка сохранения изображения: {output_path}")
            return None