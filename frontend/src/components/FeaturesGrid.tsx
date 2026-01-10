import { Scissors, Shirt, Sparkles, ShoppingBag } from "lucide-react";
import { GlowingEffect } from "@/components/ui/glowing-effect";
import { cn } from "@/lib/utils";
import "./FeaturesGrid.css";

interface GridItemProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  area?: string;
}

const GridItem = ({ icon, title, description, area }: GridItemProps) => {
  return (
    <li className={cn("min-h-[14rem] list-none", area)}>
      <div className="relative h-full rounded-[1.25rem] border-[0.75px] border-[#333333] p-2 md:rounded-[1.5rem] md:p-3">
        <GlowingEffect
          spread={40}
          glow={true}
          disabled={false}
          proximity={64}
          inactiveZone={0.01}
          borderWidth={3}
        />
        <div className="relative flex h-full flex-col justify-between gap-6 overflow-hidden rounded-xl border-[0.75px] border-[#333333] bg-[#1a1a1a] p-6 shadow-sm md:p-6">
          <div className="relative flex flex-1 flex-col justify-between gap-3">
            <div className="w-fit rounded-lg border-[0.75px] border-[#333333] bg-[#000000] p-2 text-white">
              {icon}
            </div>
            <div className="space-y-3">
              <h3 className="pt-0.5 text-xl leading-[1.375rem] font-semibold font-sans tracking-[-0.04em] md:text-2xl md:leading-[1.875rem] text-balance text-white">
                {title}
              </h3>
              <h2 className="[&_b]:md:font-semibold [&_strong]:md:font-semibold font-sans text-sm leading-[1.125rem] md:text-base md:leading-[1.375rem] text-[#cccccc]">
                {description}
              </h2>
            </div>
          </div>
        </div>
      </div>
    </li>
  );
};

export function FeaturesGrid() {
  return (
    <div className="features-grid-container">
      <h2 className="features-grid-title">Our Products</h2>
      <ul className="grid grid-cols-1 grid-rows-none gap-4 md:grid-cols-12 md:grid-rows-2 lg:gap-4">
        <GridItem
          area="md:[grid-area:1/1/2/7]"
          icon={<Scissors className="h-4 w-4" />}
          title="Clothing Extractor"
          description="Advanced AI-powered extraction system that isolates individual clothing items from photos with precision. Perfect for cataloging your wardrobe or creating product listings."
        />
        <GridItem
          area="md:[grid-area:1/7/2/13]"
          icon={<Shirt className="h-4 w-4" />}
          title="Virtual Try-On"
          description="See how clothes look on you before you buy. Our virtual try-on technology uses AI to realistically place garments on your body, helping you make confident fashion choices."
        />
        <GridItem
          area="md:[grid-area:2/1/3/7]"
          icon={<Sparkles className="h-4 w-4" />}
          title="Outfit Recommendation"
          description="Get personalized outfit suggestions powered by AI. Our system analyzes your wardrobe and suggests perfect combinations based on your style preferences and occasions."
        />
        <GridItem
          area="md:[grid-area:2/7/3/13]"
          icon={<ShoppingBag className="h-4 w-4" />}
          title="Online Outfit Thrift Shop"
          description="Buy and sell pre-loved fashion items in our sustainable marketplace. Connect with fashion enthusiasts and build a circular fashion economy while discovering unique pieces."
        />
      </ul>
    </div>
  );
}

