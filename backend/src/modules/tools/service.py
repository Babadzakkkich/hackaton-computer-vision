import zipfile
import io
import time
import os
from datetime import datetime
from typing import Dict, List
from collections import Counter

from ml.model import YOLOModel
from .schemas import (
    SingleAnalysisResponse, BatchAnalysisResponse, ImageAnalysisResult,
    DetectionItem, AnalysisConfig, AnalysisResult
)

class ToolService:
    def __init__(self):
        self.model = YOLOModel()
        self.base_output_dir = "results"
        os.makedirs(self.base_output_dir, exist_ok=True)
        self.expected_tools_count = 11

    def error_result(self, message: str) -> AnalysisResult:
        return AnalysisResult(
            status="error",
            total_detections=0,
            message=message,
            detections=[]
        )

    def create_session_dir(self, prefix: str = "batch_session") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.base_output_dir, f"{prefix}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def process_image(self, image_data: bytes, filename: str,
                      confidence_threshold: float, iou_threshold: float,
                      session_dir: str | None = None) -> ImageAnalysisResult:
        """Общий пайплайн обработки изображения"""
        detections_dict, original_image = self.model.predict(image_data, confidence_threshold, iou_threshold)
        detection_items = self.convert_to_detection_items(detections_dict)
        analysis_result = self.analyze_tool_completeness(detection_items)

        annotated_path = self.model.save_annotated_image(
            original_image, detections_dict, filename, session_dir
        )

        print(f"Сохранено аннотированное изображение: {annotated_path}")

        return ImageAnalysisResult(
            filename=filename,
            analysis_result=analysis_result,
            annotated_image_path=annotated_path
        )


    def convert_to_detection_items(self, detections_dict: List[Dict]) -> List[DetectionItem]:
        return [
            DetectionItem(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"]
            )
            for d in detections_dict
        ]


    def analyze_tool_completeness(self, detections: List[DetectionItem]) -> AnalysisResult:
        detected_classes = set()
        class_counts = Counter()
        detected_tools_names = []

        for d in detections:
            detected_classes.add(d.class_id)
            class_counts[d.class_id] += 1
            detected_tools_names.append(d.class_name)

        expected_classes = set(range(self.expected_tools_count))
        missing = expected_classes - detected_classes
        extra = detected_classes - expected_classes
        duplicates = {cid for cid, count in class_counts.items() if count > 1}

        missing_tools = [self.model.class_names[cid] for cid in missing]
        extra_tools = [self.model.class_names[cid] for cid in extra]
        duplicate_tools = [self.model.class_names[cid] for cid in duplicates]

        total_unique = len(detected_classes)

        if total_unique == self.expected_tools_count and not extra and not duplicates:
            status, message = "complete", "Полный набор: все инструменты обнаружены"
        elif total_unique == self.expected_tools_count and duplicates:
            status, message = "duplicates", f"Полный набор с дубликатами: {len(duplicate_tools)} дублируются"
        elif missing and not extra and not duplicates:
            status, message = "missing", f"Неполный набор: отсутствует {len(missing_tools)}"
        elif missing and duplicates and not extra:
            status, message = "missing_duplicates", f"Неполный набор: отсутствует {len(missing_tools)}, дублируется {len(duplicate_tools)}"
        elif extra and not missing:
            status, message = "extra", f"Лишние инструменты: {len(extra_tools)}"
        elif extra and missing:
            status, message = "mixed", f"Смешанный результат: отсутствует {len(missing_tools)}, лишних {len(extra_tools)}"
        elif duplicates:
            status, message = "duplicates_only", f"Дубликаты: {len(duplicate_tools)}"
        else:
            status, message = "unknown", f"Неопределенный результат: {total_unique} уникальных инструментов"

        if status in {"duplicates", "duplicates_only", "missing_duplicates"}:
            extra_tools.extend([f"{tool} (дубликат)" for tool in duplicate_tools])

        return AnalysisResult(
            status=status,
            total_detections=len(detections),
            expected_count=self.expected_tools_count,
            missing_tools=missing_tools,
            extra_tools=extra_tools,
            detected_tools=detected_tools_names,
            detections=detections,
            message=message
        )

    def analyze_single_image(self, image_data: bytes, filename: str = "image",
                             confidence_threshold: float = 0.5, iou_threshold: float = 0.5) -> SingleAnalysisResponse:
        try:
            image_result = self.process_image(image_data, filename, confidence_threshold, iou_threshold)

            config = AnalysisConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                annotated_image_path=image_result.annotated_image_path
            )

            return SingleAnalysisResponse(
                status="success",
                analysis_result=image_result.analysis_result,
                config=config
            )
        except Exception as e:
            return SingleAnalysisResponse(
                status="error",
                analysis_result=self.error_result(str(e)),
                config=AnalysisConfig(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
            )

    def analyze_batch_images(self, zip_data: bytes,
                             confidence_threshold: float = 0.5, iou_threshold: float = 0.5) -> BatchAnalysisResponse:
        start_time = time.time()
        try:
            zip_file = zipfile.ZipFile(io.BytesIO(zip_data))
            image_files = [f for f in zip_file.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                return BatchAnalysisResponse(status="error", total_images=0, processed_images=0,
                                             results=[], processing_time=0, summary={})

            session_dir = self.create_session_dir("batch_session")
            results, status_summary = [], Counter()

            for filename in image_files:
                try:
                    with zip_file.open(filename) as img_file:
                        image_data = img_file.read()

                    image_result = self.process_image(image_data, filename, confidence_threshold, iou_threshold, session_dir)
                    results.append(image_result)
                    status_summary[image_result.analysis_result.status] += 1

                    print(f"Обработано: {filename} - {image_result.analysis_result.status}")

                except Exception as e:
                    error_result = self.error_result(str(e))
                    results.append(ImageAnalysisResult(filename=filename, analysis_result=error_result))
                    status_summary["error"] += 1

            processing_time = round(time.time() - start_time, 2)
            config = AnalysisConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                output_directory=session_dir,
                total_annotated_images=sum(1 for r in results if r.annotated_image_path)
            )

            return BatchAnalysisResponse(
                status="completed",
                total_images=len(image_files),
                processed_images=len(results),
                results=results,
                processing_time=processing_time,
                summary=dict(status_summary),
                config=config
            )
        except Exception as e:
            return BatchAnalysisResponse(
                status="error", total_images=0, processed_images=0, results=[],
                processing_time=round(time.time() - start_time, 2), summary={}
            )


tool_service = ToolService()
