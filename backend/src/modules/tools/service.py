import zipfile
import io
import time
import os
from datetime import datetime
from ml.model import YOLOModel
from .schemas import SingleAnalysisResponse, BatchAnalysisResponse, ImageAnalysisResult, DetectionItem, AnalysisConfig

class ToolService:
    def __init__(self):
        self.model = YOLOModel()
        # Базовая директория для результатов
        self.base_output_dir = "results"
        os.makedirs(self.base_output_dir, exist_ok=True)
    
    def analyze_single_image(self, image_data: bytes, filename: str = "image", 
                           confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                           save_annotated: bool = True) -> SingleAnalysisResponse:
        """Анализирует одно изображение с настраиваемыми параметрами"""
        try:
            # Получаем предсказания от модели и исходное изображение
            detections, original_image = self.model.predict(image_data, confidence_threshold, iou_threshold)
            
            # Сохраняем аннотированное изображение
            annotated_path = None
            if save_annotated:
                annotated_path = self.model.save_annotated_image(original_image, detections, filename)
            
            # Преобразуем в нужный формат
            detection_items = [
                DetectionItem(
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    confidence=det['confidence'],
                    bbox=det['bbox']
                ) for det in detections
            ]
            
            # Простая логика определения статуса
            if detections:
                status = "success"
                message = f"Обнаружено {len(detections)} инструментов"
            else:
                status = "no_detections"
                message = "Инструменты не обнаружены"
            
            # Добавляем информацию о использованных настройках и пути к файлу
            config = AnalysisConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                annotated_image_path=annotated_path
            )
            
            return SingleAnalysisResponse(
                status=status,
                total_detections=len(detections),
                detections=detection_items,
                message=message,
                config=config
            )
            
        except Exception as e:
            return SingleAnalysisResponse(
                status="error",
                total_detections=0,
                detections=[],
                message=f"Ошибка анализа: {str(e)}"
            )
    
    def analyze_batch_images(self, zip_data: bytes, max_images: int = 100,
                           confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                           save_annotated: bool = True) -> BatchAnalysisResponse:
        """Анализирует группу изображений из ZIP-архива с настраиваемыми параметрами"""
        start_time = time.time()
        
        try:
            # Открываем ZIP-архив
            zip_file = zipfile.ZipFile(io.BytesIO(zip_data))
            
            # Получаем список файлов в архиве
            file_list = zip_file.namelist()
            
            # Фильтруем только изображения
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [
                f for f in file_list 
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_files:
                return BatchAnalysisResponse(
                    status="error",
                    total_images=0,
                    processed_images=0,
                    total_detections=0,
                    results=[],
                    processing_time=0,
                    message="В архиве не найдено изображений"
                )
            
            # Ограничиваем количество обрабатываемых изображений
            if len(image_files) > max_images:
                image_files = image_files[:max_images]
                print(f"Ограничение: будет обработано {max_images} из {len(image_files)} изображений")
            
            # Создаем директорию для этой сессии
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.base_output_dir, f"batch_session_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            print(f"Результаты будут сохранены в: {session_dir}")
            
            results = []
            total_detections = 0
            processed_count = 0
            annotated_paths = []
            
            # Обрабатываем каждое изображение
            for filename in image_files:
                try:
                    # Читаем изображение из архива
                    with zip_file.open(filename) as image_file:
                        image_data = image_file.read()
                    
                    # Получаем предсказания и изображение
                    detections, original_image = self.model.predict(image_data, confidence_threshold, iou_threshold)
                    
                    # Сохраняем аннотированное изображение
                    annotated_path = None
                    if save_annotated:
                        # Передаем session_dir чтобы все изображения сохранялись в одной директории
                        annotated_path = self.model.save_annotated_image(original_image, detections, filename, session_dir)
                        if annotated_path:
                            annotated_paths.append(annotated_path)
                    
                    # Преобразуем детекции
                    detection_items = [
                        DetectionItem(
                            class_id=det['class_id'],
                            class_name=det['class_name'],
                            confidence=det['confidence'],
                            bbox=det['bbox']
                        ) for det in detections
                    ]
                    
                    # Определяем статус
                    if detections:
                        status = "success"
                        message = f"Обнаружено {len(detections)} инструментов"
                    else:
                        status = "no_detections"
                        message = "Инструменты не обнаружены"
                    
                    # Создаем результат для этого изображения
                    image_result = ImageAnalysisResult(
                        filename=filename,
                        status=status,
                        total_detections=len(detections),
                        detections=detection_items,
                        message=message,
                        annotated_image_path=annotated_path
                    )
                    
                    results.append(image_result)
                    total_detections += len(detections)
                    processed_count += 1
                    
                    print(f"Обработано: {filename} - {len(detections)} обнаружений")
                    if annotated_path:
                        print(f"  Аннотированное изображение: {annotated_path}")
                    
                except Exception as e:
                    # Если ошибка при обработке конкретного изображения
                    error_result = ImageAnalysisResult(
                        filename=filename,
                        status="error",
                        total_detections=0,
                        detections=[],
                        message=f"Ошибка обработки: {str(e)}"
                    )
                    results.append(error_result)
                    processed_count += 1
                    print(f"Ошибка обработки {filename}: {e}")
            
            processing_time = time.time() - start_time
            
            # Статистика обработки
            successful = len([r for r in results if r.status == "success"])
            no_detections = len([r for r in results if r.status == "no_detections"])
            errors = len([r for r in results if r.status == "error"])
            
            # Добавляем информацию о использованных настройках
            config = AnalysisConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                output_directory=session_dir,
                total_annotated_images=len(annotated_paths)
            )
            
            return BatchAnalysisResponse(
                status="completed",
                total_images=len(image_files),
                processed_images=processed_count,
                total_detections=total_detections,
                results=results,
                processing_time=round(processing_time, 2),
                message=f"Обработано {processed_count} изображений: {successful} с обнаружениями, {no_detections} без обнаружений, {errors} с ошибками. Результаты в {session_dir}",
                config=config
            )
            
        except zipfile.BadZipFile:
            processing_time = time.time() - start_time
            return BatchAnalysisResponse(
                status="error",
                total_images=0,
                processed_images=0,
                total_detections=0,
                results=[],
                processing_time=round(processing_time, 2),
                message="Некорректный ZIP-архив"
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return BatchAnalysisResponse(
                status="error",
                total_images=0,
                processed_images=0,
                total_detections=0,
                results=[],
                processing_time=round(processing_time, 2),
                message=f"Ошибка обработки архива: {str(e)}"
            )

# Создаем глобальный экземпляр сервиса
tool_service = ToolService()