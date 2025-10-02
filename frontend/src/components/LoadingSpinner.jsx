const LoadingSpinner = () => {
  return (
    <div className="loading-spinner">
      <div className="spinner"></div>
      <p>Идет обработка изображений...</p>
      <p className="loading-subtitle">Это может занять несколько секунд</p>
    </div>
  )
}

export default LoadingSpinner