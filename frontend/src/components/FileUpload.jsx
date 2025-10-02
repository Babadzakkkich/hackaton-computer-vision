import React, { useRef } from 'react'

const FileUpload = ({ onFileSelect, selectedFile }) => {
  const fileInputRef = useRef(null)

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (file) {
      onFileSelect(file)
    }
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) {
      onFileSelect(file)
    }
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  return (
    <div className="file-upload">
      <h3>📁 Загрузка файлов</h3>
      
      <div
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="drop-zone-content">
          <div className="upload-icon">📤</div>
          <p>
            {selectedFile 
              ? `Выбран: ${selectedFile.name}`
              : 'Перетащите файл сюда или нажмите для выбора'
            }
          </p>
          <small>
            Поддерживаемые форматы: JPG, PNG, ZIP с изображениями
          </small>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.zip"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {selectedFile && (
        <div className="file-info">
          <strong>Информация о файле:</strong>
          <p>Имя: {selectedFile.name}</p>
          <p>Тип: {selectedFile.type || 'Не определен'}</p>
          <p>Размер: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
        </div>
      )}
    </div>
  )
}

export default FileUpload