import { Pricing } from "@/components/ui/pricing"
import Navigation from "../components/Navigation"
import Footer from "../components/Footer"
import "./Pricing.css"

const demoPlans = [
  {
    name: "STARTER",
    price: "50",
    yearlyPrice: "40",
    period: "per month",
    features: [
      "Up to 50 wardrobe items",
      "Clothing extraction & tagging",
      "Basic outfit recommendations",
      "Mobile app access",
      "Email support",
      "Standard wardrobe analytics",
    ],
    description: "Perfect for personal wardrobe management",
    buttonText: "Start Free Trial",
    href: "/sign-up",
    isPopular: false,
  },
  {
    name: "PROFESSIONAL",
    price: "99",
    yearlyPrice: "79",
    period: "per month",
    features: [
      "Unlimited wardrobe items",
      "Advanced clothing extraction",
      "AI-powered outfit recommendations",
      "Virtual try-on preview",
      "Priority support",
      "Advanced analytics & insights",
      "Outfit planning calendar",
      "Style trend tracking",
    ],
    description: "Ideal for fashion enthusiasts and stylists",
    buttonText: "Get Started",
    href: "/sign-up",
    isPopular: true,
  },
  {
    name: "ENTERPRISE",
    price: "299",
    yearlyPrice: "239",
    period: "per month",
    features: [
      "Everything in Professional",
      "Custom AI model training",
      "Dedicated account manager",
      "1-hour support response",
      "API access for integrations",
      "Team collaboration tools",
      "Custom wardrobe solutions",
      "SLA guarantee",
    ],
    description: "For fashion brands and retail businesses",
    buttonText: "Contact Sales",
    href: "/contact",
    isPopular: false,
  },
]

function PricingPage() {
  return (
    <div className="pricing-page">
      <Navigation />
      <main className="pricing-main">
        <Pricing 
          plans={demoPlans}
          title="StyleMind Pricing"
          description="Choose the perfect plan for your wardrobe management needs. All plans include AI-powered clothing extraction, tagging, and our comprehensive wardrobe management platform."
        />
      </main>
      <Footer />
    </div>
  )
}

export default PricingPage

