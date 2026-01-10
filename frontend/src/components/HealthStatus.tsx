import { useState, useEffect } from 'react'
import './HealthStatus.css'

function HealthStatus() {
  const [status, setStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking')
  const [message, setMessage] = useState('Checking server status...')

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:5000/health')
        if (response.ok) {
          const data = await response.json()
          setStatus('healthy')
          setMessage(data.message || 'Server is healthy')
        } else {
          setStatus('unhealthy')
          setMessage('Server is not responding correctly')
        }
      } catch (error) {
        setStatus('unhealthy')
        setMessage('Cannot connect to server. Make sure the Flask backend is running on port 5000.')
      }
    }

    // Initial check on mount
    checkHealth()
    
    // Check every 2 minutes (120000ms) - much less frequent
    const interval = setInterval(checkHealth, 120000)

    return () => clearInterval(interval)
  }, []) // Only run once on mount

  return (
    <div className={`health-status ${status}`}>
      <div className="health-indicator">
        {status === 'checking' && '⏳'}
        {status === 'healthy' && '✅'}
        {status === 'unhealthy' && '❌'}
      </div>
      <div className="health-text">
        <strong>Server Status:</strong> {message}
      </div>
    </div>
  )
}

export default HealthStatus

