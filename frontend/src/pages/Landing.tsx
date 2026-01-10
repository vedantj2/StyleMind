import { ShaderAnimation } from "@/components/ui/shader-lines"
import { Link } from "react-router-dom"

export default function Landing() {
  return (
    <div className="relative flex h-screen w-full flex-col items-center justify-center overflow-hidden">
      <ShaderAnimation/>
      <div className="pointer-events-none z-10 flex flex-col items-center justify-center gap-6">
        <h1 className="text-center text-7xl leading-none font-semibold tracking-tighter whitespace-pre-wrap text-white">
          Shader Lines
        </h1>
        <p className="text-center text-xl text-white/80 font-light tracking-tight max-w-2xl">
          Advanced AI-powered clothing extraction and reconstruction system
        </p>
        <div className="pointer-events-auto flex gap-4 mt-4">
          <Link
            to="/home"
            className="px-8 py-3 bg-white text-black text-sm font-medium hover:bg-white/90 transition-colors"
          >
            Learn More
          </Link>
          <Link
            to="/playground"
            className="px-8 py-3 border border-white text-white text-sm font-medium hover:bg-white/10 transition-colors"
          >
            Try It Now
          </Link>
        </div>
      </div>
    </div>
  )
}

