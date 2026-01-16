import { useState } from 'react'
import ImageUpload from '../components/ImageUpload'
import Footer from '../components/Footer'
import './Playground.css'

interface GarmentTags {
  garment_type: string
  sub_category: string
  primary_color: string
  secondary_colors: string[]
  fabric: string
  texture: string
  pattern: string
  sleeve_type: string
  fit: string
  style: string
  season: string
  gender: 'men' | 'women' | 'unisex'
  keywords: string[]
}

interface ExtractionResult {
  originalImage: string
  maskImage: string
  reconstructedImages: Array<{ 
    name: string
    image: string
    tags?: GarmentTags
  }>
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
                <div className="button-group">
                  <button onClick={handleReset} className="reset-button">
                    Upload Different Image
                  </button>
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
                      'Start Extraction'
                    )}
                  </button>
                </div>
              </div>
            </div>

            {error && (
              <div className="error-message">
                <p>{error}</p>
              </div>
            )}

            {results && (
              <div className="results-table">
                {/* First row: Original and Mask */}
                <div className="table-row">
                  <div className="table-cell image-cell">
                    <div className="cell-header">Original Image</div>
                    <div className="image-container">
                      <img src={results.originalImage} alt="Original" />
                    </div>
                  </div>
                  <div className="table-cell image-cell">
                    <div className="cell-header">Parsing Mask</div>
                    <div className="image-container">
                      <img src={results.maskImage} alt="Mask" />
                    </div>
                  </div>
                </div>

                {/* Subsequent rows: Reconstructed images and tags */}
                {results.reconstructedImages.map((item, index) => (
                  <div key={index} className="table-row">
                    <div className="table-cell image-cell">
                      <div className="cell-header">{item.name}</div>
                      <div className="image-container">
                        <img src={item.image} alt={item.name} />
                      </div>
                    </div>
                    <div className="table-cell tags-cell">
                      <div className="cell-header">Tags</div>
                      {item.tags ? (
                        <div className="tags-content">
                          <div className="tag-group">
                            <div className="tag-item">
                              <span className="tag-label">Garment Type:</span>
                              <span className="tag-value">{item.tags.garment_type}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Sub Category:</span>
                              <span className="tag-value">{item.tags.sub_category}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Primary Color:</span>
                              <span className="tag-value">{item.tags.primary_color}</span>
                            </div>
                            {item.tags.secondary_colors && item.tags.secondary_colors.length > 0 && (
                              <div className="tag-item">
                                <span className="tag-label">Secondary Colors:</span>
                                <span className="tag-value">{item.tags.secondary_colors.join(', ')}</span>
                              </div>
                            )}
                          </div>
                          <div className="tag-group">
                            <div className="tag-item">
                              <span className="tag-label">Fabric:</span>
                              <span className="tag-value">{item.tags.fabric}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Texture:</span>
                              <span className="tag-value">{item.tags.texture}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Pattern:</span>
                              <span className="tag-value">{item.tags.pattern}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Sleeve Type:</span>
                              <span className="tag-value">{item.tags.sleeve_type}</span>
                            </div>
                          </div>
                          <div className="tag-group">
                            <div className="tag-item">
                              <span className="tag-label">Fit:</span>
                              <span className="tag-value">{item.tags.fit}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Style:</span>
                              <span className="tag-value">{item.tags.style}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Season:</span>
                              <span className="tag-value">{item.tags.season}</span>
                            </div>
                            <div className="tag-item">
                              <span className="tag-label">Gender:</span>
                              <span className="tag-value">{item.tags.gender}</span>
                            </div>
                          </div>
                          {item.tags.keywords && item.tags.keywords.length > 0 && (
                            <div className="tag-group">
                              <div className="tag-item full-width">
                                <span className="tag-label">Keywords:</span>
                                <div className="keywords-list">
                                  {item.tags.keywords.map((keyword, idx) => (
                                    <span key={idx} className="keyword-badge">{keyword}</span>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="tags-content no-tags">
                          <p>Tags not available</p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <Footer />
    </div>
  )
}

export default Playground
