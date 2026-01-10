import { cn } from "@/lib/utils";
import { useState } from "react";
import { Mail, Lock } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // Temporary login credentials
  const TEMP_EMAIL = "demo@stylemind.com";
  const TEMP_PASSWORD = "demo123";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Validate credentials
    if (email === TEMP_EMAIL && password === TEMP_PASSWORD) {
      // Store login state (you can use localStorage or context later)
      if (rememberMe) {
        localStorage.setItem("rememberMe", "true");
      }
      localStorage.setItem("isLoggedIn", "true");
      // Navigate to dashboard
      navigate("/dashboard");
      // You can add more logic here like storing auth token, etc.
    } else {
      setError("Invalid email or password. Use demo@stylemind.com / demo123");
    }
  };

  return (
    <div className="flex h-[700px] w-full">
      <div className="w-full hidden md:inline-block">
        <img 
          className="h-full w-full object-cover" 
          src="https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&h=1200&fit=crop" 
          alt="Fashion background" 
        />
      </div>

      <div className="w-full flex flex-col items-center justify-center bg-black">
        <form onSubmit={handleSubmit} className="md:w-96 w-80 flex flex-col items-center justify-center">
          <h2 className="text-4xl text-white font-medium">Sign in</h2>
          <p className="text-sm text-white/60 mt-3">Welcome back! Please sign in to continue</p>

          <button 
            type="button" 
            className="w-full mt-8 bg-white/10 border border-white/20 flex items-center justify-center h-12 rounded-full hover:bg-white/20 transition-colors"
          >
            <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            <span className="text-white">Continue with Google</span>
          </button>

          <div className="flex items-center gap-4 w-full my-5">
            <div className="w-full h-px bg-white/20"></div>
            <p className="w-full text-nowrap text-sm text-white/60">or sign in with email</p>
            <div className="w-full h-px bg-white/20"></div>
          </div>

          <div className="flex items-center w-full bg-transparent border border-white/20 h-12 rounded-full overflow-hidden pl-6 gap-2">
            <Mail className="w-4 h-4 text-white/60" />
            <input 
              type="email" 
              placeholder="Email id" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="bg-transparent text-white placeholder-white/60 outline-none text-sm w-full h-full" 
              required 
            />                 
          </div>

          <div className="flex items-center mt-6 w-full bg-transparent border border-white/20 h-12 rounded-full overflow-hidden pl-6 gap-2">
            <Lock className="w-4 h-4 text-white/60" />
            <input 
              type="password" 
              placeholder="Password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="bg-transparent text-white placeholder-white/60 outline-none text-sm w-full h-full" 
              required 
            />
          </div>

          {error && (
            <div className="w-full mt-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
              {error}
            </div>
          )}

          <div className="w-full flex items-center justify-between mt-8 text-white/60">
            <div className="flex items-center gap-2">
              <input 
                className="h-5 w-5" 
                type="checkbox" 
                id="checkbox" 
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <label className="text-sm cursor-pointer" htmlFor="checkbox">Remember me</label>
            </div>
            <a className="text-sm underline hover:text-white transition-colors" href="#">Forgot password?</a>
          </div>

          <button 
            type="submit" 
            className="mt-8 w-full h-11 rounded-full text-black bg-white hover:bg-white/90 transition-opacity font-medium"
          >
            Login
          </button>
          
          <div className="mt-4 p-3 bg-white/5 border border-white/10 rounded-lg text-white/60 text-xs text-center">
            Demo credentials: <br />
            Email: <span className="text-white font-mono">demo@stylemind.com</span><br />
            Password: <span className="text-white font-mono">demo123</span>
          </div>
          <p className="text-white/60 text-sm mt-4">Don't have an account? <a className="text-white hover:underline" href="#">Sign up</a></p>
        </form>
      </div>
    </div>
  );
}

