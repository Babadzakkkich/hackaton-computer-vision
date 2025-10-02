import React, { useRef, useEffect } from 'react'

const ImageSlider = ({ results, currentIndex, onIndexChange }) => {
  const sliderRef = useRef(null)
  const activeThumbnailRef = useRef(null)

  useEffect(() => {
    if (activeThumbnailRef.current && sliderRef.current) {
      const slider = sliderRef.current
      const activeElement = activeThumbnailRef.current
      
      const sliderRect = slider.getBoundingClientRect()
      const activeRect = activeElement.getBoundingClientRect()
      
      // Auto-scroll to keep active thumbnail visible
      if (activeRect.left < sliderRect.left) {
        slider.scrollLeft -= (sliderRect.left - activeRect.left + 10)
      } else if (activeRect.right > sliderRect.right) {
        slider.scrollLeft += (activeRect.right - sliderRect.right + 10)
      }
    }
  }, [currentIndex])

  // Функция для получения иконки статуса
  const getStatusIcon = (result) => {
    const status = result.analysis_result?.status || result.status
    
    switch (status) {
      case 'complete': return '✅'
      case 'missing': return '⚠️'
      case 'extra': return '❌'
      case 'mixed': return '🔄'
      case 'duplicates': return '🔁'
      case 'duplicates_only': return '🔁'
      case 'missing_duplicates': return '🔄'
      case 'error': return '❌'
      default: return '❓'
    }
  }

  // Функция для получения количества обнаружений
  const getDetectionsCount = (result) => {
    return result.analysis_result?.total_detections || result.total_detections || 0
  }

  return (
    <div className="image-slider">
      <div className="slider-controls">
        <button
          onClick={() => onIndexChange(Math.max(0, currentIndex - 1))}
          disabled={currentIndex === 0}
          className="slider-button"
        >
          ◀️
        </button>
        
        <div className="slider-info">
          <div className="slider-main-info">
            Изображение {currentIndex + 1} из {results.length}
          </div>
          {results[currentIndex]?.filename && (
            <div className="filename">
              {results[currentIndex].filename}
            </div>
          )}
        </div>

        <button
          onClick={() => onIndexChange(Math.min(results.length - 1, currentIndex + 1))}
          disabled={currentIndex === results.length - 1}
          className="slider-button"
        >
          ▶️
        </button>
      </div>

      <div className="slider-container" ref={sliderRef}>
        <div className="slider-track">
          {results.map((result, index) => (
            <div
              key={index}
              ref={index === currentIndex ? activeThumbnailRef : null}
              className={`slider-thumbnail ${index === currentIndex ? 'active' : ''}`}
              onClick={() => onIndexChange(index)}
            >
              <div className="thumbnail-status">
                {getStatusIcon(result)}
              </div>
              <div className="thumbnail-info">
                {getDetectionsCount(result)} об.
              </div>
              <div className="thumbnail-number">
                {index + 1}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default ImageSlider