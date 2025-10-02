import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import yaml
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch

class YOLOModel:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = self._setup_device()
        self.load_model()
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.font = self._load_font()
    
    def _setup_device(self):
        try:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def _load_font(self):
        font_paths = [
            "/app/fonts/ComicRelief-Regular.ttf",
            "fonts/ComicRelief-Regular.ttf", 
            "../fonts/ComicRelief-Regular.ttf",
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, 24)
        return ImageFont.load_default()
    
    def load_model(self):
        model_paths = [
            "/app/weights/best.pt",
            "weights/best.pt", 
            "../weights/best.pt",
        ]
        
        data_yaml_paths = [
            "/app/data/data.yaml",
            "data/data.yaml",
            "../data/data.yaml", 
        ]
        
        model_path = next((p for p in model_paths if os.path.exists(p)), None)
        data_yaml_path = next((p for p in data_yaml_paths if os.path.exists(p)), None)
        
        if not model_path:
            raise FileNotFoundError("Файл модели не найден")
        if not data_yaml_path:
            raise FileNotFoundError("Файл data.yaml не найден")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        with open(data_yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            self.class_names = data.get('names', [])
    
    def predict(self, image_data: bytes, confidence_threshold: float = 0.25, iou_threshold: float = 0.45) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Не удалось декодировать изображение")
        
        results = self.model.predict(
            image,
            conf=confidence_threshold,
            iou=iou_threshold,
            agnostic_nms=True,
            max_det=15,
            device=self.device,
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
    
    def get_device_info(self) -> Dict[str, Any]:
        info = {
            "device": self.device,
            "device_type": "GPU" if self.device.startswith("cuda") else "CPU",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if self.device.startswith("cuda"):
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                info.update({
                    "gpu_name": gpu_props.name,
                    "gpu_memory_gb": gpu_props.total_memory / 1024**3,
                    "gpu_count": torch.cuda.device_count(),
                })
            except Exception as e:
                info["gpu_error"] = str(e)
        
        return info

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        colors = [
            (46, 204, 113), (52, 152, 219), (231, 76, 60),
            (155, 89, 182), (241, 196, 15), (26, 188, 156),
            (230, 126, 34), (149, 165, 166), (52, 73, 94),
            (127, 140, 141), (44, 62, 80)
        ]
        
        text_bg_colors = [
            (39, 174, 96), (41, 128, 185), (192, 57, 43),
            (142, 68, 173), (243, 156, 18), (22, 160, 133),
            (211, 84, 0), (127, 140, 141), (44, 62, 80),
            (108, 122, 137), (36, 50, 71)
        ]
        
        for detection in detections:
            class_id = detection['class_id']
            confidence = detection['confidence']
            bbox = detection['bbox']
            class_name = detection['class_name']
            
            box_color = colors[class_id % len(colors)]
            text_bg_color = text_bg_colors[class_id % len(text_bg_colors)]
            x1, y1, x2, y2 = map(int, bbox)
            
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            label = f"{class_name}: {confidence:.2f}"
            bbox_text = draw.textbbox((0, 0), label, font=self.font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            padding = 8
            bg_x1 = max(0, x1 - padding // 2)
            bg_y1 = max(0, y1 - text_height - 10 - padding // 2)
            bg_x2 = x1 + text_width + padding
            bg_y2 = y1 + padding // 2
            
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=text_bg_color)
            
            text_x = x1
            text_y = y1 - text_height - 5
            
            draw.text((text_x + 1, text_y + 1), label, fill=(0, 0, 0), font=self.font)
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=self.font)
            
            corner_size = 8
            draw.line([x1, y1, x1 + corner_size, y1], fill=box_color, width=2)
            draw.line([x1, y1, x1, y1 + corner_size], fill=box_color, width=2)
            draw.line([x2, y1, x2 - corner_size, y1], fill=box_color, width=2)
            draw.line([x2, y1, x2, y1 + corner_size], fill=box_color, width=2)
            draw.line([x1, y2, x1 + corner_size, y2], fill=box_color, width=2)
            draw.line([x1, y2, x1, y2 - corner_size], fill=box_color, width=2)
            draw.line([x2, y2, x2 - corner_size, y2], fill=box_color, width=2)
            draw.line([x2, y2, x2, y2 - corner_size], fill=box_color, width=2)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def save_annotated_image(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                           filename: str, output_dir: str = None) -> str:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"session_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        annotated_image = self.draw_detections(image, detections)
        clean_filename = os.path.basename(filename)
        output_path = os.path.join(output_dir, f"annotated_{clean_filename}")
        success = cv2.imwrite(output_path, annotated_image)
        
        return output_path if success else None