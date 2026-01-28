import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navigation from './components/Navigation'
import Home from './pages/Home'
import Pricing from './pages/Pricing'
import Playground from './pages/Playground'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Profile from './pages/Profile'
import OutfitRecommendations from './pages/OutfitRecommendations'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/pricing" element={<Pricing />} />
          <Route path="/playground" element={
            <>
              <Navigation />
              <main className="app-main">
                <Playground />
              </main>
            </>
          } />
          <Route path="/login" element={<Login />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/dashboard/profile" element={<Profile />} />
          <Route path="/dashboard/outfit-recommendations" element={<OutfitRecommendations />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App
