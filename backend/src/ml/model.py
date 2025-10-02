import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import yaml
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

class YOLOModel:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.load_model()
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Загружаем единый шрифт для всех систем
        self.font = self._load_font()
    
    def _load_font(self):
        """Загружает единый шрифт из папки fonts"""
        try:
            # Пробуем разные возможные пути к шрифту
            font_paths = [
                "/app/fonts/ComicRelief-Regular.ttf",  # Абсолютный путь в контейнере
                "fonts/ComicRelief-Regular.ttf",       # Относительный путь
                "../fonts/ComicRelief-Regular.ttf",    # Относительный путь из src
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 24)
                    print(f"✅ Шрифт загружен: {font_path}")
                    return font
            
            # Если шрифт не найден, используем стандартный
            print("⚠️  Шрифт ComicRelief-Regular.ttf не найден, используется стандартный шрифт")
            return ImageFont.load_default()
            
        except Exception as e:
            print(f"⚠️  Ошибка загрузки шрифта: {e}, используется стандартный шрифт")
            return ImageFont.load_default()
    
    def load_model(self):
        """Загружает модель и классы из YAML"""
        try:
            model_path = "weights/best.pt"
            data_yaml_path = "data/data.yaml"
            
            # Проверяем существование файлов
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"Файл data.yaml не найден: {data_yaml_path}")
            
            print(f"Загружаем модель из: {model_path}")
            self.model = YOLO(model_path)
            
            with open(data_yaml_path, 'r', encoding='utf-8') as file:
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
        """Рисует bounding boxes на изображении с улучшенной читаемостью"""
        # Конвертируем BGR в RGB для PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Улучшенная цветовая палитра
        colors = [
            (46, 204, 113),   # Спокойный зеленый
            (52, 152, 219),   # Приятный синий
            (231, 76, 60),    # Неяркий красный
            (155, 89, 182),   # Фиолетовый
            (241, 196, 15),   # Золотистый
            (26, 188, 156),   # Бирюзовый
            (230, 126, 34),   # Оранжевый
            (149, 165, 166),  # Серый
            (52, 73, 94),     # Темно-синий
            (127, 140, 141),  # Серо-голубой
            (44, 62, 80)      # Темно-серый
        ]
        
        # Более тёмные цвета для фона текста
        text_bg_colors = [
            (39, 174, 96),    # Тёмный зеленый
            (41, 128, 185),   # Тёмный синий
            (192, 57, 43),    # Тёмный красный
            (142, 68, 173),   # Тёмный фиолетовый
            (243, 156, 18),   # Тёмный золотистый
            (22, 160, 133),   # Тёмный бирюзовый
            (211, 84, 0),     # Тёмный оранжевый
            (127, 140, 141),  # Серый
            (44, 62, 80),     # Тёмно-синий
            (108, 122, 137),  # Тёмный серо-голубой
            (36, 50, 71)      # Очень тёмный серый
        ]
        
        for detection in detections:
            class_id = detection['class_id']
            confidence = detection['confidence']
            bbox = detection['bbox']
            class_name = detection['class_name']
            
            # Выбираем цвета для класса
            box_color = colors[class_id % len(colors)]
            text_bg_color = text_bg_colors[class_id % len(text_bg_colors)]
            
            # Координаты bounding box
            x1, y1, x2, y2 = map(int, bbox)
            
            # Рисуем прямоугольник с увеличенной толщиной
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            # Текст с названием класса и уверенностью
            label = f"{class_name}: {confidence:.2f}"
            
            # Получаем размеры текста
            bbox_text = draw.textbbox((0, 0), label, font=self.font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Добавляем отступы к фону текста
            padding = 8
            bg_x1 = max(0, x1 - padding // 2)
            bg_y1 = max(0, y1 - text_height - 10 - padding // 2)
            bg_x2 = x1 + text_width + padding
            bg_y2 = y1 + padding // 2
            
            # Фон для текста
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=text_bg_color)
            
            # Текст с белым цветом
            text_x = x1
            text_y = y1 - text_height - 5
            
            # Добавляем тень для текста
            draw.text((text_x + 1, text_y + 1), label, fill=(0, 0, 0), font=self.font)
            
            # Основной текст
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=self.font)
            
            # Рисуем уголки для визуального выделения
            corner_size = 8
            # Левый верхний угол
            draw.line([x1, y1, x1 + corner_size, y1], fill=box_color, width=2)
            draw.line([x1, y1, x1, y1 + corner_size], fill=box_color, width=2)
            # Правый верхний угол
            draw.line([x2, y1, x2 - corner_size, y1], fill=box_color, width=2)
            draw.line([x2, y1, x2, y1 + corner_size], fill=box_color, width=2)
            # Левый нижний угол
            draw.line([x1, y2, x1 + corner_size, y2], fill=box_color, width=2)
            draw.line([x1, y2, x1, y2 - corner_size], fill=box_color, width=2)
            # Правый нижний угол
            draw.line([x2, y2, x2 - corner_size, y2], fill=box_color, width=2)
            draw.line([x2, y2, x2, y2 - corner_size], fill=box_color, width=2)
        
        # Конвертируем обратно в BGR для OpenCV
        image_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image_with_boxes
    
    def save_annotated_image(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                           filename: str, output_dir: str = None) -> str:
        """Сохраняет изображение с bounding boxes"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"session_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        annotated_image = self.draw_detections(image, detections)
        clean_filename = os.path.basename(filename)
        output_path = os.path.join(output_dir, f"annotated_{clean_filename}")
        success = cv2.imwrite(output_path, annotated_image)
        
        if success:
            print(f"Изображение сохранено: {output_path}")
            return output_path
        else:
            print(f"Ошибка сохранения изображения: {output_path}")
            return None