import { useState } from 'react'
import './ResultsDisplay.css'

interface ResultsDisplayProps {
  images: string[]
  zipUrl: string
}

function ResultsDisplay({ images, zipUrl }: ResultsDisplayProps) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = zipUrl
    link.download = 'reconstructed_clothing.zip'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="results-display">
      <div className="results-header">
        <h2>Reconstruction Results</h2>
        <button onClick={handleDownload} className="download-button">
          📥 Download All ({images.length} images)
        </button>
      </div>

      {images.length === 0 ? (
        <div className="no-results">
          <p>No images found in the result ZIP file.</p>
        </div>
      ) : (
        <div className="results-grid">
          {images.map((imageUrl, index) => (
            <div
              key={index}
              className="result-item"
              onClick={() => setSelectedImage(imageUrl)}
            >
              <img src={imageUrl} alt={`Reconstructed item ${index + 1}`} />
              <div className="result-overlay">
                <span>Click to view</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedImage && (
        <div className="image-modal" onClick={() => setSelectedImage(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedImage(null)}>
              ×
            </button>
            <img src={selectedImage} alt="Full size" />
          </div>
        </div>
      )}
    </div>
  )
}

export default ResultsDisplay

