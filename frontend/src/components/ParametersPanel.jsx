import React from 'react'

const ParametersPanel = ({ parameters, onChange }) => {
  const handleChange = (key, value) => {
    onChange({
      ...parameters,
      [key]: value
    })
  }

  return (
    <div className="parameters-panel">
      <h3>⚙️ Параметры модели</h3>
      
      <div className="parameter">
        <label htmlFor="confidence">
          Уверенность (confidence): {parameters.confidence}
        </label>
        <input
          id="confidence"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={parameters.confidence}
          onChange={(e) => handleChange('confidence', parseFloat(e.target.value))}
        />
        <div className="parameter-help">
          Более высокое значение = меньше ложных срабатываний, но можно пропустить объекты
        </div>
      </div>

      <div className="parameter">
        <label htmlFor="iou">
          Пересечение (IoU): {parameters.iou}
        </label>
        <input
          id="iou"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={parameters.iou}
          onChange={(e) => handleChange('iou', parseFloat(e.target.value))}
        />
        <div className="parameter-help">
          Порог для подавления дублирующих обнаружений
        </div>
      </div>

      <div className="parameters-info">
        <h4>Рекомендации:</h4>
        <ul>
          <li>📷 Для четких изображений: confidence = 0.5-0.7</li>
          <li>🌅 Для сложных условий: confidence = 0.2-0.4</li>
          <li>⚡ Баланс скорости/точности: IoU = 0.4-0.6</li>
          <li>💾 Аннотированные изображения сохраняются автоматически</li>
        </ul>
      </div>
    </div>
  )
}

export default ParametersPanel