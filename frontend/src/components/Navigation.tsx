import { Link, useLocation } from 'react-router-dom'
import './Navigation.css'

interface NavigationProps {
  className?: string
}

function Navigation({ className }: NavigationProps) {
  const location = useLocation()
  const isHomePage = location.pathname === '/'
  const navClassName = className || (isHomePage ? "navbar" : "navbar navbar-not-sticky")

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
          <Link 
            to="/playground" 
            className={`navbar-link ${location.pathname === '/playground' ? 'active' : ''}`}
          >
            Playground
          </Link>
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

