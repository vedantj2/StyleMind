import { WebGLShader } from '@/components/ui/web-gl-shader'
import { LiquidButton } from '@/components/ui/button'
import Navigation from '../components/Navigation'
import Footer from '../components/Footer'
import { FeaturesGrid } from '../components/FeaturesGrid'
import './Home.css'

function Home() {
  return (
    <div className="home">
      <Navigation />
      <div className="home-hero">
        <WebGLShader />
        <div className="hero-content">
          <h1 className="mb-3 text-white text-center text-4xl font-extrabold tracking-tighter md:text-[clamp(2rem,5vw,4rem)] uppercase">
            LEVEL UP YOUR FASHION STANDARD
          </h1>
          <p className="text-white/60 px-6 text-center text-xs md:text-sm lg:text-lg">
            AI powered wardrobe management system
          </p>
          <div className="flex justify-center mt-8">
            <LiquidButton className="text-white border rounded-full" size={'xl'}>
              Let's Go
            </LiquidButton>
          </div>
        </div>
      </div>

      <FeaturesGrid />

      <div className="home-content">
        <section className="info-section">
          <h2>How It Works</h2>
          <div className="steps-container">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Image Parsing</h3>
                <p>
                  Upload an image containing a person wearing clothing. Our SCHP (Self-Correction for Human Parsing) 
                  model analyzes the image and generates a detailed segmentation mask identifying different 
                  clothing items and body parts.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Clothing Extraction</h3>
                <p>
                  The system extracts individual clothing items from the original image using the parsing mask. 
                  Each item (hat, shirt, pants, shoes, etc.) is isolated with a transparent background, 
                  ready for further processing.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Garment Cleaning</h3>
                <p>
                  OpenCV-based processing removes black backgrounds, crops tightly to the garment, and applies 
                  morphological operations to clean up the extracted clothing items, ensuring optimal quality 
                  for reconstruction.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">4</div>
              <div className="step-content">
                <h3>AI Reconstruction</h3>
                <p>
                  Using OpenAI's advanced image models, each clothing item is reconstructed as a high-quality 
                  ecommerce product image. The AI lays garments flat, removes wrinkles, preserves textures 
                  and colors, and creates professional product photography with white backgrounds.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="info-section">
          <h2>Supported Clothing Items</h2>
          <div className="items-grid">
            <div className="item-card">Hat</div>
            <div className="item-card">Gloves</div>
            <div className="item-card">Sunglasses</div>
            <div className="item-card">Upper Clothes</div>
            <div className="item-card">Dress</div>
            <div className="item-card">Coat</div>
            <div className="item-card">Socks</div>
            <div className="item-card">Pants</div>
            <div className="item-card">Jumpsuits</div>
            <div className="item-card">Scarf</div>
            <div className="item-card">Left Shoe</div>
            <div className="item-card">Right Shoe</div>
          </div>
        </section>

        <section className="info-section">
          <h2>API Endpoints</h2>
          <div className="api-info">
            <div className="api-endpoint">
              <code className="method get">GET</code>
              <code className="endpoint">/health</code>
              <p>Check API health status</p>
            </div>
            <div className="api-endpoint">
              <code className="method post">POST</code>
              <code className="endpoint">/reconstruct</code>
              <p>Upload image and get reconstructed clothing items</p>
            </div>
          </div>
        </section>
      </div>
      <Footer />
    </div>
  )
}

export default Home

