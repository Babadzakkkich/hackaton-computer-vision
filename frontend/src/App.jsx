import React, { useState } from 'react'
import FileUpload from './components/FileUpload'
import ParametersPanel from './components/ParametersPanel'
import ResultsDisplay from './components/ResultsDisplay'
import LoadingSpinner from './components/LoadingSpinner'
import { analyzeSingleImage, analyzeBatchImages } from './services/api'

function App() {
  const [file, setFile] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [parameters, setParameters] = useState({
    confidence: 0.25,
    iou: 0.45,
  })

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile)
    setResults(null)
    setError(null)
  }

  const handleParametersChange = (newParameters) => {
    setParameters(newParameters)
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError('Пожалуйста, выберите файл')
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      let response
      if (file.type === 'application/zip' || file.name.toLowerCase().endsWith('.zip')) {
        response = await analyzeBatchImages(file, parameters)
      } else {
        response = await analyzeSingleImage(file, parameters)
      }
      setResults(response)
    } catch (err) {
      setError(err.message || 'Произошла ошибка при обработке')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🔧 Сервис распознавания инструментов</h1>
        <p>Автоматическое обнаружение инструментов с помощью компьютерного зрения</p>
      </header>

      <div className="app-container">
        <div className="left-panel">
          <ParametersPanel
            parameters={parameters}
            onChange={handleParametersChange}
          />
          
          <FileUpload
            onFileSelect={handleFileSelect}
            selectedFile={file}
          />

          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={!file || isLoading}
          >
            {isLoading ? 'Обработка...' : 'Начать анализ'}
          </button>

          {error && (
            <div className="error-message">
              ❌ {error}
            </div>
          )}
        </div>

        <div className="right-panel">
          {isLoading && <LoadingSpinner />}
          
          {results && !isLoading && (
            <ResultsDisplay
              results={results}
              fileType={file?.type}
            />
          )}

          {!results && !isLoading && (
            <div className="welcome-message">
              <h2>Добро пожаловать!</h2>
              <p>Загрузите изображение или ZIP-архив с инструментами для анализа.</p>
              <div className="features">
                <div className="feature">
                  <strong>📷 Одиночное изображение</strong>
                  <p>Загрузите одно фото с инструментами</p>
                </div>
                <div className="feature">
                  <strong>📦 Пакетная обработка</strong>
                  <p>Загрузите ZIP-архив с несколькими изображениями</p>
                </div>
                <div className="feature">
                  <strong>⚙️ Настройка параметров</strong>
                  <p>Регулируйте чувствительность модели</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App