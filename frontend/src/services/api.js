const API_BASE_URL = 'http://localhost:8000'

export const analyzeSingleImage = async (file, parameters) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const queryParams = new URLSearchParams({
    confidence: parameters.confidence,
    iou: parameters.iou,
  })

  const response = await fetch(`${API_BASE_URL}/tools/analyze?${queryParams}`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || 'Ошибка при анализе изображения')
  }

  return await response.json()
}

export const analyzeBatchImages = async (file, parameters) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const queryParams = new URLSearchParams({
    confidence: parameters.confidence,
    iou: parameters.iou,
  })

  const response = await fetch(`${API_BASE_URL}/tools/analyze-batch?${queryParams}`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || 'Ошибка при анализе архива')
  }

  return await response.json()
}

export const getImageUrl = (imagePath) => {
  if (!imagePath) return null
  
  if (imagePath.startsWith('http') || imagePath.startsWith('data:')) {
    return imagePath
  }
  
  let relativePath = imagePath
  if (imagePath.includes('results/')) {
    relativePath = imagePath.split('results/')[1]
  } else if (imagePath.includes('results\\')) {
    relativePath = imagePath.split('results\\')[1]
  }
  
  return `${API_BASE_URL}/tools/images/${relativePath}`
}

export const fetchImage = async (imagePath) => {
  try {
    const url = getImageUrl(imagePath)
    const response = await fetch(url)
    
    if (!response.ok) {
      throw new Error('Ошибка загрузки изображения')
    }
    
    const blob = await response.blob()
    return URL.createObjectURL(blob)
  } catch (error) {
    console.error('Ошибка загрузки изображения:', error)
    return null
  }
}