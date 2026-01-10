import { WebGLShader } from '@/components/ui/web-gl-shader'
import { LiquidButton } from '@/components/ui/button'
import Navigation from '../components/Navigation'
import Footer from '../components/Footer'
import { FeaturesGrid } from '../components/FeaturesGrid'
import { Testimonials } from '../components/ui/testimonials'
import { useNavigate } from 'react-router-dom'
import './Home.css'

function Home() {
  const navigate = useNavigate()

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
            <LiquidButton 
              className="text-white border rounded-full" 
              size={'xl'}
              onClick={() => navigate('/login')}
            >
              Let's Go
            </LiquidButton>
          </div>
        </div>
      </div>

      <FeaturesGrid />

      <div className="home-content">
        <section className="info-section">
          <h2>Our Mission</h2>
          <div className="steps-container">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Wardrobe Management Chaos</h3>
                <p>
                  Many people struggle to organize and catalog their clothing items. We solve this by providing 
                  AI-powered tools that automatically extract and catalog clothing from photos, making wardrobe 
                  management effortless and digital.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Online Shopping Uncertainty</h3>
                <p>
                  Buying clothes online often leads to uncertainty about fit, style, and appearance. Our virtual 
                  try-on technology helps users visualize how clothes will look on them before making a purchase, 
                  reducing returns and increasing confidence in online shopping decisions.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Outfit Planning Challenges</h3>
                <p>
                  Creating stylish and appropriate outfits can be time-consuming and overwhelming. Our AI-powered 
                  outfit recommendation system analyzes your wardrobe and suggests perfect combinations based on 
                  your style preferences, occasions, and current trends.
                </p>
              </div>
            </div>

            <div className="step">
              <div className="step-number">4</div>
              <div className="step-content">
                <h3>Sustainable Fashion Access</h3>
                <p>
                  Fast fashion contributes to environmental waste, while sustainable options are often hard to find 
                  or expensive. Our thrift marketplace connects fashion enthusiasts, making it easy to buy and sell 
                  pre-loved items, promoting a circular fashion economy and making sustainable fashion more accessible.
                </p>
              </div>
            </div>
          </div>
        </section>

        <Testimonials />
      </div>
      <Footer />
    </div>
  )
}

export default Home

