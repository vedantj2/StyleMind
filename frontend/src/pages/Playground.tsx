import { useState } from 'react'
import ImageUpload from '../components/ImageUpload'
import Footer from '../components/Footer'
import './Playground.css'

interface ExtractionResult {
  originalImage: string
  maskImage: string
  reconstructedImages: Array<{ name: string; image: string }>
}

function Playground() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<ExtractionResult | null>(null)

  const handleImageUpload = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string)
      setUploadedFile(file)
      setResults(null)
      setError(null)
    }
    reader.readAsDataURL(file)
  }

  const handleExtract = async () => {
    if (!uploadedFile) return

    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const response = await fetch('http://localhost:5000/reconstruct', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error occurred' }))
        throw new Error(errorData.error || `Server error: ${response.status}`)
      }

      // Parse JSON response
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setUploadedImage(null)
    setUploadedFile(null)
    setResults(null)
    setError(null)
  }

  return (
    <div className="playground">
      <div className="playground-header">
        <h1>Clothing Extraction Playground</h1>
        <p>Upload an image and extract clothing items using AI</p>
      </div>

      <div className="playground-content">
        {!uploadedImage ? (
          <div className="upload-section">
            <ImageUpload 
              onUpload={handleImageUpload} 
              loading={false}
              disabled={false}
            />
          </div>
        ) : (
          <div className="extraction-flow">
            <div className="image-preview-section">
              <div className="preview-card">
                <h3>Uploaded Image</h3>
                <div className="image-container">
                  <img src={uploadedImage} alt="Uploaded" />
                </div>
                <button onClick={handleReset} className="reset-button">
                  Upload Different Image
                </button>
              </div>

              <div className="action-section">
                <button 
                  onClick={handleExtract} 
                  className="extract-button"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner-small"></span>
                      Processing...
                    </>
                  ) : (
                    <>
                      <span>🚀</span>
                      Start Extraction
                    </>
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="error-message">
                <p>❌ {error}</p>
              </div>
            )}

            {results && (
              <>
                <div className="comparison-section">
                  <div className="comparison-card">
                    <h3>Original Image</h3>
                    <div className="image-container">
                      <img src={results.originalImage} alt="Original" />
                    </div>
                  </div>
                  <div className="comparison-card">
                    <h3>Parsing Mask</h3>
                    <div className="image-container">
                      <img src={results.maskImage} alt="Mask" />
                    </div>
                  </div>
                </div>

                {results.reconstructedImages.length > 0 && (
                  <div className="results-section">
                    <h2>Reconstructed Clothing Items</h2>
                    <div className="results-grid">
                      {results.reconstructedImages.map((item, index) => (
                        <div key={index} className="result-card">
                          <div className="result-image-container">
                            <img src={item.image} alt={item.name} />
                          </div>
                          <div className="result-label">{item.name}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
      <Footer />
    </div>
  )
}

export default Playground

