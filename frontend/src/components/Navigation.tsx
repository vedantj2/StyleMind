import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { ChevronDown, Scissors, Shirt, Sparkles, ShoppingBag } from 'lucide-react'
import './Navigation.css'

interface NavigationProps {
  className?: string
}

interface PlaygroundFeature {
  name: string
  path: string
  icon: React.ReactNode
  available: boolean
}

const playgroundFeatures: PlaygroundFeature[] = [
  {
    name: 'Clothing Extractor',
    path: '/playground',
    icon: <Scissors className="h-4 w-4" />,
    available: true
  },
  {
    name: 'Virtual Try-On',
    path: '/playground',
    icon: <Shirt className="h-4 w-4" />,
    available: false
  },
  {
    name: 'Outfit Recommendation',
    path: '/playground',
    icon: <Sparkles className="h-4 w-4" />,
    available: false
  },
  {
    name: 'Online Outfit Thrift Shop',
    path: '/playground',
    icon: <ShoppingBag className="h-4 w-4" />,
    available: false
  }
]

function Navigation({ className }: NavigationProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const isHomePage = location.pathname === '/'
  const navClassName = className || (isHomePage ? "navbar" : "navbar navbar-not-sticky")
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!isDropdownOpen) return

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node
      if (
        dropdownRef.current && 
        !dropdownRef.current.contains(target) &&
        !(target instanceof Element && target.closest('.navbar-dropdown-menu'))
      ) {
        setIsDropdownOpen(false)
      }
    }

    // Use a small delay to avoid immediate closure
    const timeoutId = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside)
    }, 0)

    return () => {
      clearTimeout(timeoutId)
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isDropdownOpen])

  const handleFeatureClick = (feature: PlaygroundFeature) => {
    if (feature.available) {
      navigate(feature.path)
      setIsDropdownOpen(false)
    }
  }

  const isPlaygroundActive = location.pathname === '/playground'
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0, width: 0 })

  // Calculate dropdown position
  useEffect(() => {
    if (isDropdownOpen && dropdownRef.current) {
      const rect = dropdownRef.current.getBoundingClientRect()
      setDropdownPosition({
        top: rect.bottom + 8,
        left: rect.left + rect.width / 2,
        width: rect.width
      })
    }
  }, [isDropdownOpen])

  return (
    <nav className={navClassName}>
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <img 
            src="/logo.png" 
            alt="Clothing AI" 
            className="logo-image"
          />
        </Link>
        <div className="navbar-menu">
          <Link 
            to="/" 
            className={`navbar-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            Home
          </Link>
          <Link 
            to="/pricing" 
            className={`navbar-link ${location.pathname === '/pricing' ? 'active' : ''}`}
          >
            Pricing
          </Link>
          <div 
            ref={dropdownRef}
            className="navbar-dropdown"
          >
            <button
              className={`navbar-link navbar-dropdown-toggle ${isPlaygroundActive ? 'active' : ''}`}
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            >
              Playground
              <ChevronDown className={`dropdown-icon ${isDropdownOpen ? 'open' : ''}`} />
            </button>
            {isDropdownOpen && createPortal(
              <div 
                className="navbar-dropdown-menu"
                style={{
                  top: `${dropdownPosition.top}px`,
                  left: `${dropdownPosition.left}px`,
                  transform: 'translateX(-50%)'
                }}
              >
                {playgroundFeatures.map((feature, index) => (
                  <button
                    key={index}
                    className={`dropdown-item ${feature.available ? 'available' : 'coming-soon'}`}
                    onClick={() => handleFeatureClick(feature)}
                    disabled={!feature.available}
                  >
                    <span className="dropdown-item-icon">{feature.icon}</span>
                    <span className="dropdown-item-text">{feature.name}</span>
                    {!feature.available && (
                      <span className="dropdown-item-badge">Coming Soon</span>
                    )}
                  </button>
                ))}
              </div>,
              document.body
            )}
          </div>
          <Link 
            to="/login" 
            className={`navbar-link ${location.pathname === '/login' ? 'active' : ''}`}
          >
            Login
          </Link>
        </div>
      </div>
    </nav>
  )
}

export default Navigation

