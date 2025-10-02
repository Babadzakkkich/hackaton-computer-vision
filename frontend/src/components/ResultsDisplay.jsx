import { useState, useEffect } from 'react'
import ImageSlider from './ImageSlider'
import { getImageUrl} from '../services/api'

const ResultsDisplay = ({ results, fileType }) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [imageUrls, setImageUrls] = useState({})

  const isBatch = fileType === 'application/zip' || results.results?.length > 1

  const getCurrentResult = () => {
    if (isBatch && results.results) {
      return results.results[currentImageIndex]
    }
    return results
  }

  const currentResult = getCurrentResult()
  const analysisResult = currentResult.analysis_result || currentResult

  const getAnnotatedImagePath = () => {
    if (isBatch && currentResult.annotated_image_path) {
      return currentResult.annotated_image_path
    }
    if (!isBatch && currentResult.config?.annotated_image_path) {
      return currentResult.config.annotated_image_path
    }
    return null
  }

  const annotatedImagePath = getAnnotatedImagePath()

  useEffect(() => {
    const loadImages = async () => {
      const urls = {}
      
      if (isBatch && results.results) {
        for (let i = 0; i < results.results.length; i++) {
          const result = results.results[i]
          const path = result.annotated_image_path
          if (path) {
            const url = getImageUrl(path)
            urls[path] = url
          }
        }
      } else if (!isBatch && annotatedImagePath) {
        const url = getImageUrl(annotatedImagePath)
        urls[annotatedImagePath] = url
      }
      
      setImageUrls(urls)
    }

    loadImages()
  }, [results, isBatch, annotatedImagePath])

  const getImageUrlForDisplay = () => {
    if (isBatch && currentResult.annotated_image_path) {
      return imageUrls[currentResult.annotated_image_path] || getImageUrl(currentResult.annotated_image_path)
    }
    if (!isBatch && annotatedImagePath) {
      return imageUrls[annotatedImagePath] || getImageUrl(annotatedImagePath)
    }
    return null
  }

  const imageUrl = getImageUrlForDisplay()

  const getStatusColor = (status) => {
    switch (status) {
      case 'complete': return '#28a745'
      case 'missing': return '#ffc107'
      case 'extra': return '#dc3545'
      case 'mixed': return '#fd7e14'
      case 'duplicates': return '#17a2b8'
      case 'duplicates_only': return '#17a2b8'
      case 'missing_duplicates': return '#fd7e14'
      case 'error': return '#6c757d'
      default: return '#6c757d'
    }
  }

  return (
    <div className="results-display">
      <div className="results-header">
        <h2>📊 Анализ полноты набора инструментов</h2>
        {isBatch && (
          <div className="batch-info">
            Обработано изображений: {results.results?.length || 0}
          </div>
        )}
      </div>

      {isBatch && results.results && (
        <ImageSlider
          results={results.results}
          currentIndex={currentImageIndex}
          onIndexChange={setCurrentImageIndex}
        />
      )}

      <div className="results-content-vertical">
        <div className="image-section-vertical">
          {imageUrl ? (
            <div className="annotated-image-vertical">
              <img 
                src={imageUrl}
                alt="Аннотированное изображение"
                onError={(e) => {
                  console.error('Ошибка загрузки изображения:', imageUrl)
                  e.target.style.display = 'none'
                  const placeholder = e.target.parentElement.querySelector('.image-fallback')
                  if (placeholder) placeholder.style.display = 'block'
                }}
              />
              <div className="image-fallback" style={{display: 'none'}}>
                <div className="image-placeholder-vertical">
                  <p>🖼️ Изображение обработано и сохранено</p>
                  <p className="image-path">Путь: {annotatedImagePath}</p>
                  <p className="image-note">
                    Для просмотра изображения откройте файл по указанному пути
                  </p>
                </div>
              </div>
            </div>
          ) : annotatedImagePath ? (
            <div className="image-info-vertical">
              <div className="image-placeholder-vertical">
                <p>🖼️ Изображение обработано и сохранено</p>
                <p className="image-path">Путь: {annotatedImagePath}</p>
                <p className="image-note">
                  <a href={getImageUrl(annotatedImagePath)} target="_blank" rel="noopener noreferrer">
                    Открыть изображение в новой вкладке
                  </a>
                </p>
              </div>
            </div>
          ) : (
            <div className="no-image-vertical">
              <div className="no-image-content-vertical">
                <p>📷 Изображение не содержит обнаруженных объектов</p>
                <p className="no-image-note">
                  Модель не нашла инструментов на этом изображении
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="info-section-vertical">
          <div className="info-grid">
            {/* Статус анализа */}
            <div 
              className="status-badge"
              style={{ backgroundColor: getStatusColor(analysisResult.status) }}
            >
              <div className="status-message">
                {analysisResult.message || analysisResult.status}
              </div>
              <div className="status-details">
                Обнаружено: {analysisResult.total_detections} из {analysisResult.expected_count || 11} инструментов
              </div>
            </div>

            {analysisResult.missing_tools && analysisResult.missing_tools.length > 0 && (
              <div className="missing-tools-section">
                <h4>❌ Отсутствующие инструменты:</h4>
                <div className="tools-list missing">
                  {analysisResult.missing_tools.map((tool, index) => (
                    <div key={index} className="tool-item missing">
                      {tool}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {analysisResult.extra_tools && analysisResult.extra_tools.length > 0 && (
              <div className="extra-tools-section">
                <h4>⚠️ Лишние инструменты:</h4>
                <div className="tools-list extra">
                  {analysisResult.extra_tools.map((tool, index) => (
                    <div key={index} className="tool-item extra">
                      {tool}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {analysisResult.detections && analysisResult.detections.length > 0 && (
              <div className="detections-list">
                <h4>🎯 Обнаруженные инструменты:</h4>
                {analysisResult.detections.map((detection, index) => (
                  <div key={index} className="detection-item">
                    <span className="tool-name">{detection.class_name}</span>
                    <span className="confidence">
                      Уверенность: {(detection.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {isBatch && results.summary && (
              <div className="batch-summary">
                <h4>📈 Сводка по всем изображениям:</h4>
                <div className="summary-grid">
                  <div className="summary-item complete">
                    <span>Полные наборы:</span>
                    <span className="count">{results.summary.complete || 0}</span>
                  </div>
                  <div className="summary-item missing">
                    <span>Неполные наборы:</span>
                    <span className="count">{results.summary.missing || 0}</span>
                  </div>
                  <div className="summary-item duplicates">
                    <span>С дубликатами:</span>
                    <span className="count">{(results.summary.duplicates || 0) + (results.summary.duplicates_only || 0) + (results.summary.missing_duplicates || 0)}</span>
                  </div>
                  <div className="summary-item error">
                    <span>Ошибки:</span>
                    <span className="count">{results.summary.error || 0}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultsDisplay