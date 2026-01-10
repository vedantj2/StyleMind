import LoginForm from '../components/ui/login-form'
import Navigation from '../components/Navigation'
import Footer from '../components/Footer'
import './Login.css'

function Login() {
  return (
    <div className="login-page">
      <Navigation />
      <main className="login-main">
        <LoginForm />
      </main>
      <Footer />
    </div>
  )
}

export default Login

