import { TestimonialsColumn } from "@/components/ui/testimonials-columns-1";
import { motion } from "motion/react";

const testimonials = [
  {
    text: "StyleMind revolutionized my wardrobe management. The AI extraction feature helped me catalog my entire closet in minutes. No more wondering what I own!",
    image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&h=400&fit=crop",
    name: "Sarah Chen",
    role: "Fashion Enthusiast",
  },
  {
    text: "The virtual try-on feature is a game-changer for online shopping. I can finally see how clothes will look on me before buying. Reduced my returns by 80%!",
    image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop",
    name: "Michael Rodriguez",
    role: "E-commerce Manager",
  },
  {
    text: "As someone who struggles with outfit planning, the AI recommendations are incredible. It suggests perfect combinations I never would have thought of.",
    image: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&h=400&fit=crop",
    name: "Emily Johnson",
    role: "Marketing Director",
  },
  {
    text: "The thrift marketplace is amazing! I've found so many unique pieces and sold items I no longer wear. It's sustainable fashion made easy.",
    image: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop",
    name: "David Kim",
    role: "Sustainability Advocate",
  },
  {
    text: "StyleMind's clothing extraction is incredibly accurate. I use it to create product listings for my online boutique. Saves me hours of work!",
    image: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&h=400&fit=crop",
    name: "Jessica Martinez",
    role: "Boutique Owner",
  },
  {
    text: "The outfit recommendation system understands my style perfectly. It's like having a personal stylist available 24/7. Love it!",
    image: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400&h=400&fit=crop",
    name: "Alex Thompson",
    role: "Style Blogger",
  },
  {
    text: "I've tried many wardrobe apps, but StyleMind is by far the best. The AI-powered features are intuitive and actually work as promised.",
    image: "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400&h=400&fit=crop",
    name: "Ryan Patel",
    role: "Tech Professional",
  },
  {
    text: "The virtual try-on feature helped me build confidence in my online shopping. I know exactly what will look good before it arrives.",
    image: "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?w=400&h=400&fit=crop",
    name: "Olivia Brown",
    role: "Fashion Consultant",
  },
  {
    text: "StyleMind has transformed how I manage my wardrobe. The extraction feature makes it so easy to organize and the recommendations are spot-on.",
    image: "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=400&h=400&fit=crop",
    name: "James Wilson",
    role: "Creative Director",
  },
];

const firstColumn = testimonials.slice(0, 3);
const secondColumn = testimonials.slice(3, 6);
const thirdColumn = testimonials.slice(6, 9);

export const Testimonials = () => {
  return (
    <section className="bg-black my-20 relative">
      <div className="container z-10 mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          viewport={{ once: true }}
          className="flex flex-col items-center justify-center max-w-[540px] mx-auto"
        >
          <div className="flex justify-center">
            <div className="border border-white/30 text-white py-1 px-4 rounded-lg">Testimonials</div>
          </div>

          <h2 className="text-white text-xl sm:text-2xl md:text-3xl lg:text-4xl xl:text-5xl font-bold tracking-tighter mt-5">
            What our users say
          </h2>
          <p className="text-center mt-5 text-white/60">
            See what our customers have to say about us.
          </p>
        </motion.div>

        <div className="flex justify-center gap-6 mt-10 [mask-image:linear-gradient(to_bottom,transparent,black_25%,black_75%,transparent)] max-h-[740px] overflow-hidden">
          <TestimonialsColumn testimonials={firstColumn} duration={15} />
          <TestimonialsColumn testimonials={secondColumn} className="hidden md:block" duration={19} />
          <TestimonialsColumn testimonials={thirdColumn} className="hidden lg:block" duration={17} />
        </div>
      </div>
    </section>
  );
};

