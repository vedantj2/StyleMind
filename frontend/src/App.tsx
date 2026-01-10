import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navigation from './components/Navigation'
import Home from './pages/Home'
import Pricing from './pages/Pricing'
import Playground from './pages/Playground'
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
        </Routes>
      </div>
    </Router>
  )
}

export default App
